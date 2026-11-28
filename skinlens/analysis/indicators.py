"""
八項 VISIA 式指標的影像辨識 (傳統電腦視覺版，不需要訓練資料就能跑)。

每個指標輸出：
  raw      原始量測值 (例如斑點面積%、皺紋長度密度) → 給訓練模型當特徵
  severity 0~100 嚴重度 (用 logistic 函數把 raw 映射過去；參數是經驗值，之後用資料庫標註校正)
  count    偵測到的病灶數
  mask     要疊在臉上顯示的遮罩
  by_region 各分區的量測值

一般手機照片 (可見光) 只能做前 6 項；紫外線色斑、紫質需要上傳 UV 光源照片 (lighting="uv")。
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from skimage.filters import frangi
from . import chromophore, wrinkle_seg
from skimage.measure import label, regionprops

# 疊圖顏色 (BGR)
COLORS = {
    "spots": (40, 40, 220), "wrinkles": (255, 200, 0), "texture": (0, 200, 255), "pores": (255, 0, 200),
    "uv_spots": (200, 120, 40), "brown_spots": (20, 90, 160), "red_areas": (60, 60, 255), "porphyrins": (0, 255, 255),
}

# logistic 參數 (mid, scale)：raw = mid 時 severity = 50
CALIB = {
    "spots": (0.9, 0.35),        # 斑點面積 %
    "wrinkles": (6.0, 2.5),      # 每千像素皺紋長度
    "texture": (5.0, 1.2),       # 高頻殘差標準差
    "pores": (12.0, 5.0),        # 每萬像素毛孔數
    "uv_spots": (3.0, 1.2),      # UV 暗斑面積 %
    "brown_spots": (2.5, 1.0),   # 棕色斑面積 %
    "red_areas": (6.0, 2.5),     # 泛紅面積 %
    "porphyrins": (8.0, 3.0),    # 每萬像素紫質螢光點
}

# 用參考族群校準過的參數 (scripts/calibrate.py 產生) 覆蓋上面的經驗值
_CALIB_FILE = Path(__file__).resolve().parent.parent.parent / "models" / "calib.json"
if _CALIB_FILE.exists():
    CALIB.update({k: tuple(v) for k, v in json.loads(_CALIB_FILE.read_text(encoding="utf-8"))["calib"].items()})

# AI 皺紋分割的計分：和 FFHQ-Wrinkle 人工答案完全相同的換算 (scripts/import_ffhq_wrinkle.py 產生)
_WRINKLE_LABEL_FILE = _CALIB_FILE.parent / "wrinkle_label.json"
WRINKLE_SEG_CALIB = None
if _WRINKLE_LABEL_FILE.exists():
    _w = json.loads(_WRINKLE_LABEL_FILE.read_text(encoding="utf-8"))
    WRINKLE_SEG_CALIB = (_w["mid"], _w["scale"], _w["eps"])


# 紅色區 (RBX 式) 的門檻：單位是「高出全臉血紅素基準幾個標準差 (MAD 換算)」
RED_Z_DIFFUSE = 2.0     # 大片泛紅 (平滑後)
RED_Z_FOCAL = 3.0       # 局部紅點 (相對周圍)
# 斑點 (色素圖的局部偏高)：周圍範圍 σ (px，臉寬 600)、門檻 z、雜訊下限 (黑色素, 血紅素；光學密度單位)
SPOT_SIGMA = 10.0
SPOT_Z = 3.0
SPOT_FLOOR = (0.022, 0.011)   # 約手機照片的局部雜訊 (FFHQ 網路照片壓縮雜訊較大，第 5 百分位才到這裡)
SPOT_MIN_SOLIDITY = 0.75
SPOT_ROI = ("forehead", "glabella", "nose", "cheek_l", "cheek_r")   # VISIA 斑點的分析範圍

def logistic(x, mid, scale):
    return float(100 / (1 + np.exp(-(x - mid) / scale)))


def _filter_components(binary, min_area, max_area, max_ecc=None, min_ecc=None, min_len=None, min_solidity=None):
    lab = label(binary > 0)
    keep = []
    for r in regionprops(lab):
        if not (min_area <= r.area <= max_area):
            continue
        if min_solidity is not None and r.solidity < min_solidity:   # 月牙形、不規則形 (鼻翼溝、陰影邊緣)
            continue
        if max_ecc is not None and r.eccentricity > max_ecc:
            continue
        if min_ecc is not None and r.eccentricity < min_ecc:
            continue
        if min_len is not None and r.axis_major_length < min_len:
            continue
        keep.append(r.label)
    return np.isin(lab, keep), len(keep)


def _core(pp, k=11):
    """侵蝕過的皮膚遮罩：避開臉部邊緣、髮際、五官交界的陰影，減少誤判。"""
    return cv2.erode(pp["skin"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def _by_region(mask, regions, fn):
    res = {}
    for k, rm in regions.items():
        area = int((rm > 0).sum())
        if area > 200:
            res[k] = round(fn(mask & (rm > 0), rm > 0, area), 3)
    return res


def _area_pct(m, rm, area):
    return 100 * m.sum() / area


def _pack(name, raw, count, mask, regions, by_region_fn=_area_pct, extra=None, calib=None):
    mid, sc, *log_eps = calib or CALIB[name]
    x = np.log(raw + log_eps[0]) if log_eps else raw     # 校準檔可指定在對數尺度上計算 (右偏分布)
    return {"raw": round(float(raw), 3), "severity": round(logistic(x, mid, sc), 1), "count": int(count),
            "mask": mask, "by_region": _by_region(mask, regions, by_region_fn), **(extra or {})}


# ------------------------------------------------------------------------ 可見光指標
def _local_z(img, valid, sigma, floor):
    """色素圖的「局部偏高」：減掉周圍的平均 (只用有效皮膚做加權模糊，五官、反光不會被抹進來)，
    再除以自己臉上的雜訊 (MAD 換算的標準差，至少 floor，避免很乾淨的臉把微小起伏也放大成斑點)。"""
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(np.where(valid, img, 0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)
    local = img - num / np.maximum(den, 1e-3)
    med, sig = chromophore.robust_stats(local[valid])
    return (local - med) / max(sig, floor)


def _spot_exclude(pp):
    """斑點偵測額外排除的五官交界 (天生色差、陰影、鬍根，不是斑點)：
      鼻翼與鼻孔下緣  鼻翼溝天生較紅、鼻孔邊有陰影、男生還有鬍根
      嘴角            唇緣偏紅、嘴角紋路的陰影、鬍根
      內眼角          淚阜的粉紅色、鼻樑側面的陰影"""
    kp, ipd = pp["kp"], pp["ipd"]
    cx = (kp["eye_l"][0] + kp["eye_r"][0]) / 2
    ny = kp["nose"][1]
    m = np.zeros(pp["skin"].shape, np.uint8)
    cv2.rectangle(m, (int(cx - 0.32 * ipd), int(ny - 0.08 * ipd)), (int(cx + 0.32 * ipd), int(ny + 0.22 * ipd)), 1, -1)
    for k in ("mouth_l", "mouth_r"):
        cv2.circle(m, (int(kp[k][0]), int(kp[k][1])), int(0.2 * ipd), 1, -1)
    polys = pp.get("polys")
    for k in ("eye_l", "eye_r"):
        if polys:
            p = polys[k]
            inner = p[np.argmin(np.abs(p[:, 0] - cx))]      # 眼睛輪廓最靠近鼻樑的點 = 內眼角
        else:
            e = kp[k]
            inner = (e[0] + np.sign(cx - e[0]) * 0.25 * ipd, e[1])
        cv2.circle(m, (int(inner[0]), int(inner[1])), int(0.16 * ipd), 1, -1)
    if polys:   # 唇緣再多留一點距離
        lips = np.zeros_like(m)
        cv2.fillPoly(lips, [polys["lips"].astype(np.int32)], 1)
        m |= cv2.dilate(lips, np.ones((int(0.14 * ipd) | 1,) * 2, np.uint8))
    return m > 0


def _spot_roi(pp):
    """斑點的分析範圍 (VISIA 總覽畫面的青色框：額頭 + 鼻子 + 兩頰)。"""
    m = np.zeros(pp["skin"].shape, bool)
    for k in SPOT_ROI:
        m |= pp["regions"][k] > 0
    return m


def detect_spots(pp):
    """斑點：和周圍膚色有明顯顏色差異的棕色或紅色小斑塊 (曬斑、雀斑、痣、痘疤、痘印)。仿 VISIA：
      1. 色素分離 (chromophore.py)：黑色素圖 (棕) + 血紅素圖 (紅)。光學密度裡光照強弱是三通道等量變化，
         被分到第三個成分 → 等於已經做了光照平整，陰影、皺紋凹陷、毛孔的暗影不會被當成斑點
      2. 兩張圖各自找「比周圍明顯偏高」的地方 (局部 z > SPOT_Z)
      3. 只在 VISIA 的 ROI 找 (額頭、眉間、鼻子、兩頰；SPOT_ROI)：眼角、黑眼圈、嘴角、下巴的
         天生色差和鬍根不算進來
      4. 留下小而不細長的塊 (排除皺紋、毛髮) → 依哪種色素偏高較多分成棕色 / 紅色斑點
      5. 量化：斑點面積占 ROI 有效皮膚 %、個數 (棕色 / 紅色)
    沒有色素方向檔時退回舊方法 (光照平整後的 L 通道黑帽運算)。"""
    mh = chromophore.maps(pp)
    skin = _core(pp)
    if mh is None or skin.sum() < 500:
        return _detect_spots_lab(pp)
    valid = skin & ~chromophore.invalid_mask(pp, skin) & ~_spot_exclude(pp)
    roi = valid & _spot_roi(pp)
    if roi.sum() < 500:
        roi = valid
    mel = cv2.GaussianBlur(mh[0], (0, 0), 1.0)    # 壓掉感光雜訊 / JPEG 區塊
    hem = cv2.GaussianBlur(mh[1], (0, 0), 1.0)
    zm = _local_z(mel, valid, SPOT_SIGMA, SPOT_FLOOR[0])
    zh = _local_z(hem, valid, SPOT_SIGMA, SPOT_FLOOR[1])
    # 周圍基準與雜訊用整張臉的有效皮膚算，只在 ROI 裡找斑點
    cand = ((zm > SPOT_Z) | (zh > SPOT_Z)) & roi
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    # 斑點多半是實心的圓塊；鼻翼溝 (天生較紅) 的月牙形、陰影邊緣的不規則形用實心度排除
    mask, n = _filter_components(cand, 10, 900, max_ecc=0.93, min_solidity=SPOT_MIN_SOLIDITY)
    lab_img = label(mask)
    n_brown = n_red = 0
    for r in regionprops(lab_img):
        sel = lab_img == r.label
        if zm[sel].mean() >= zh[sel].mean():
            n_brown += 1
        else:
            n_red += 1
    raw = 100 * mask.sum() / max(int(roi.sum()), 1)
    return _pack("spots", raw, n, mask, pp["regions"],
                 extra={"brown_count": n_brown, "red_count": n_red, "method": "chromophore"})


def _detect_spots_lab(pp):
    """舊方法：比周圍皮膚暗的小色塊。光照平整後的 L 通道黑帽運算，再排除細長的 (皺紋 / 毛髮)。"""
    skin = _core(pp)
    L = cv2.cvtColor(pp["face_flat"], cv2.COLOR_BGR2LAB)[..., 0]   # 光照平整後：陰影側的斑點對比不會被壓低
    bh =cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    cand = (bh > 13) & skin
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    mask, n = _filter_components(cand, 8, 900, max_ecc=0.93)
    raw = 100 * mask.sum() / max(skin.sum(), 1)
    return _pack("spots", raw, n, mask, pp["regions"], extra={"method": "lab"})


def detect_brown_spots(pp):
    """棕色斑：黑色素沉澱 (肝斑、曬斑、發炎後色素沉澱)。比周圍暗 + 偏黃棕 (Lab 的 b* 較高)。"""
    skin = _core(pp, 15)
    lab = cv2.cvtColor(pp["face_flat"], cv2.COLOR_BGR2LAB).astype(np.float32)   # 光照平整後
    L, b = lab[..., 0], lab[..., 2]
    Ld = L - cv2.GaussianBlur(L, (0, 0), 15)
    bd = b - cv2.GaussianBlur(b, (0, 0), 15)
    cand = (Ld < -7) & (bd > 2) & skin
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mask, n = _filter_components(cand, 15, 5000, max_ecc=0.95)
    raw = 100 * mask.sum() / max(skin.sum(), 1)
    unevenness = float(np.std(b[skin])) if skin.any() else 0
    return _pack("brown_spots", raw, n, mask, pp["regions"], extra={"pigment_unevenness": round(unevenness, 2)})


def detect_red_areas(pp):
    """紅色區：血管擴張/發炎 (泛紅、酒糟、痘痘發炎、敏感)。仿 VISIA RBX 的流程：
      1. 反光排除 (VISIA 用交叉偏振光消除表面油光；手機沒有偏振片，改成偵測反光像素後排除)
      2. 色素分離：線性空間白平衡 → 光學密度 → 血紅素圖 (chromophore.py，Tsumura 1999；VISIA 的 R 通道)
         黑色素與陰影被分到其他成分，紅黑交雜的痘印也只取血紅素
      3. 以自己整張臉的血紅素基準 (中位數、MAD) 找三種異常：
           大片泛紅  平滑後高於全臉基準 (酒糟、敏感泛紅)
           局部紅點  明顯高於周圍 (發炎丘疹、痘痘)
           血管線條  血紅素圖上的亮脊線 (擴張的微血管)
      4. 量化：面積 % (raw)、紅點數 (count)、強度 (異常區平均高出基準幾個標準差)
    沒有色素方向檔 (models/chromophore.json) 時退回 Lab a* 的舊方法。"""
    mh = chromophore.maps(pp)
    skin = _core(pp)
    if mh is None or skin.sum() < 500:
        return _detect_red_areas_lab(pp)
    valid = skin & ~chromophore.invalid_mask(pp, skin)
    hemo = cv2.GaussianBlur(mh[1], (0, 0), 1.2)             # 壓掉感光雜訊 / JPEG 區塊
    base, sig = chromophore.robust_stats(hemo[valid])
    z = (hemo - base) / sig                                   # 高出全臉基準幾個標準差

    # 大片泛紅：只和自己的臉比。不用族群的絕對門檻：手機照片光源不受控，白平衡後殘留的色偏
    # 會投影到血紅素方向，暖光下整張臉都被判成泛紅 (VISIA 能直接比是因為燈光固定)。
    # 整張臉偏紅的程度留給特徵 chromo.hemoglobin_mean，由校正模型從人工評分學。
    diffuse = (cv2.GaussianBlur(z, (0, 0), 6) > RED_Z_DIFFUSE) & valid
    diffuse, _ = _filter_components(diffuse, 80, 10 ** 7)

    local = hemo - cv2.GaussianBlur(hemo, (0, 0), 12)
    lb, ls = chromophore.robust_stats(local[valid])
    focal_c = (((local - lb) / ls) > RED_Z_FOCAL) & (z > 1.0) & valid
    focal_c = cv2.morphologyEx(focal_c.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    papules, n_pap = _filter_components(focal_c, 12, 400, max_ecc=0.9)

    fr = frangi(np.clip(z, -3, 8) / 8, sigmas=(1.0, 1.5, 2.0), black_ridges=False)
    fr = fr / (fr[valid].max() + 1e-9)
    vessels, _ = _filter_components((fr > 0.15) & (z > 1.0) & valid, 10, 3000, min_ecc=0.95, min_len=12)

    mask = diffuse | papules | vessels
    n_valid = max(int(valid.sum()), 1)
    raw = 100 * mask.sum() / n_valid
    intensity = float(z[mask].mean()) if mask.any() else 0.0
    return _pack("red_areas", raw, n_pap, mask, pp["regions"],
                 extra={"inflamed_spots": int(n_pap), "intensity": round(intensity, 3),
                        "diffuse_pct": round(100 * diffuse.sum() / n_valid, 3),
                        "vessel_pct": round(100 * vessels.sum() / n_valid, 3), "method": "rbx"})


def _detect_red_areas_lab(pp):
    """舊方法：Lab 的 a* 明顯高於自己臉部的中位數。"""
    skin = _core(pp)
    a = cv2.cvtColor(pp["face"], cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    if not skin.any():
        return _pack("red_areas", 0, 0, np.zeros_like(skin), pp["regions"])
    ad = a - np.median(a[skin])
    cand = (ad > 7) & skin
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    mask, _ = _filter_components(cand, 20, 10 ** 7)
    # 小而集中的紅點 → 可能是發炎性痘痘
    papules, n_pap = _filter_components((ad > 11) & skin, 12, 400, max_ecc=0.9)
    raw = 100 * mask.sum() / skin.sum()
    return _pack("red_areas", raw, n_pap, mask, pp["regions"], extra={"inflamed_spots": int(n_pap), "method": "lab"})


def detect_wrinkles(pp):
    """皺紋：有 AI 模型 (U-Net) 就用它畫出的皺紋線計分，否則用傳統 Frangi 濾波。"""
    if wrinkle_seg.available() and WRINKLE_SEG_CALIB is not None:
        return _detect_wrinkles_unet(pp)
    return _detect_wrinkles_frangi(pp)


def _detect_wrinkles_unet(pp):
    """U-Net 畫出皺紋線 → 皺紋量 = 線條像素 ÷ 臉寬² (和人工答案同公式、同換算)。
    FFHQ-Wrinkle 測試組：Dice 0.618 (Frangi 0.108)，皺紋量和人工相關 0.87。"""
    reg = pp["regions"]
    mask = wrinkle_seg.predict_mask(pp["face_unet"])   # U-Net 用訓練時的舊版校色輸入
    kp = pp["kp"]
    face_w = max(float(kp["face_r"][0] - kp["face_l"][0]), 1.0)
    raw = float(mask.sum()) / face_w ** 2
    n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return _pack("wrinkles", raw, n - 1, mask, reg, calib=WRINKLE_SEG_CALIB,
                 by_region_fn=lambda m, rm, area: 1000 * m.sum() / area, extra={"method": "unet"})


def _detect_wrinkles_frangi(pp):
    """傳統方法：Frangi 脊線濾波找暗色細長線條，只在易長皺紋的區域 (額頭、眉間、魚尾、眼下、法令紋)。"""
    reg = pp["regions"]
    zone = np.zeros(pp["skin"].shape, bool)
    for k in ("forehead", "glabella", "crow_l", "crow_r", "under_eye", "nasolabial"):
        zone |= reg[k] > 0
    g = cv2.GaussianBlur(pp["gray_clahe"], (3, 3), 0).astype(np.float32) / 255
    fr = frangi(g, sigmas=(1.5, 2.5, 3.5), black_ridges=True)
    fr = fr / (fr[zone].max() + 1e-9) if zone.any() else fr
    cand = (fr > 0.12) & zone
    mask, n = _filter_components(cand, 25, 10 ** 6, min_ecc=0.96, min_len=22)
    # 用骨架長度當皺紋長度
    from skimage.morphology import skeletonize
    skel = skeletonize(mask)
    raw = 1000 * skel.sum() / max(zone.sum(), 1)
    return _pack("wrinkles", raw, n, mask, reg, extra={"method": "frangi"},
                 by_region_fn=lambda m, rm, area: 1000 * skeletonize(m).sum() / area)


def detect_texture(pp):
    """紋理：皮膚表面平滑度。高通濾波後的殘差越大 = 越粗糙 (角質堆積、凹凸、痘疤)。"""
    skin = _core(pp, 15)
    # 光照平整後：亮的地方殘差大、暗的地方小，不平整的話亮度差會被當成粗糙度
    g = cv2.cvtColor(pp["face_flat"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 4)
    if not skin.any():
        return _pack("texture", 0, 0, np.zeros_like(skin), pp["regions"])
    raw = float(np.std(hp[skin]))
    local = np.sqrt(np.maximum(cv2.blur(hp ** 2, (15, 15)), 0))
    mask = (local > max(6.0, raw * 1.6)) & skin
    return _pack("texture", raw, 0, mask, pp["regions"],
                 by_region_fn=lambda m, rm, area: float(np.std(hp[rm])))


def detect_pores(pp):
    """毛孔：鼻子、臉頰上規則分佈的小圓形暗點。小核黑帽 + 面積/圓度篩選。"""
    reg = pp["regions"]
    zone = (reg["nose"] > 0) | (reg["cheek_l"] > 0) | (reg["cheek_r"] > 0) | (reg["forehead"] > 0)
    g = pp["gray_clahe"]
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cand = (bh > 12) & zone
    mask, n = _filter_components(cand, 3, 60, max_ecc=0.9)
    raw = 10000 * n / max(zone.sum(), 1)
    return _pack("pores", raw, n, mask, reg,
                 by_region_fn=lambda m, rm, area: 10000 * label(m).max() / area)


# ------------------------------------------------------------------------ UV 光源指標
def detect_uv_spots(pp):
    """紫外線色斑：表皮下的日曬傷害，肉眼看不到，UV 光下會變成暗斑。需上傳 UV 照片。"""
    skin = pp["skin"] > 0
    g = cv2.cvtColor(pp["face"], cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    mask, n = _filter_components((bh > 15) & skin, 15, 3000, max_ecc=0.95)
    return _pack("uv_spots", 100 * mask.sum() / max(skin.sum(), 1), n, mask, pp["regions"])


def detect_porphyrins(pp):
    """紫質：痤瘡桿菌代謝物，UV 光下發橘紅色螢光，常聚集在毛孔(T 字部位)。需上傳 UV 照片。"""
    skin = pp["skin"] > 0
    f = pp["face"].astype(np.int16)
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    th = cv2.morphologyEx(pp["face"][..., 2], cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8))
    cand = (R - B > 35) & (R - G > 10) & (th > 18) & skin
    mask, n = _filter_components(cand, 2, 80)
    return _pack("porphyrins", 10000 * n / max(skin.sum(), 1), n, mask, pp["regions"],
                 by_region_fn=lambda m, rm, area: 10000 * label(m).max() / area)


VISIBLE = {"spots": detect_spots, "wrinkles": detect_wrinkles, "texture": detect_texture,
           "pores": detect_pores, "brown_spots": detect_brown_spots, "red_areas": detect_red_areas}
UV = {"uv_spots": detect_uv_spots, "porphyrins": detect_porphyrins}


def analyze_all(pp, lighting="standard"):
    fns = UV if lighting == "uv" else VISIBLE
    return {k: f(pp) for k, f in fns.items()}


def overlay(face, mask, color, alpha=0.65):
    out = face.copy()
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    # 外框讓小點也看得到
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 1)
    return out


def feature_vector(results, pp):
    """把所有量測值攤平成一個 dict，給機器學習模型當輸入特徵。"""
    feats = {}
    for k, r in results.items():
        feats[f"{k}.raw"] = r["raw"]
        feats[f"{k}.count"] = r["count"]
        for reg, v in r["by_region"].items():
            feats[f"{k}.{reg}"] = v
        for extra in ("pigment_unevenness", "inflamed_spots", "intensity", "diffuse_pct", "vessel_pct",
                      "brown_count", "red_count"):
            if extra in r:
                feats[f"{k}.{extra}"] = r[extra]
    skin = pp["skin"] > 0
    if skin.any():
        lab = cv2.cvtColor(pp["face"], cv2.COLOR_BGR2LAB)[skin].astype(np.float32)
        for i, c in enumerate("Lab"):
            feats[f"color.{c}_mean"] = float(lab[:, i].mean())
            feats[f"color.{c}_std"] = float(lab[:, i].std())
    feats.update({f"quality.{k}": v for k, v in pp["quality"].items() if isinstance(v, (int, float))})
    feats.update(chromophore.features(pp))     # 黑色素 / 血紅素分離與反光 (Tsumura et al. 1999)
    return feats
