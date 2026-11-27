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


def logistic(x, mid, scale):
    return float(100 / (1 + np.exp(-(x - mid) / scale)))


def _filter_components(binary, min_area, max_area, max_ecc=None, min_ecc=None, min_len=None):
    lab = label(binary > 0)
    keep = []
    for r in regionprops(lab):
        if not (min_area <= r.area <= max_area):
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
def detect_spots(pp):
    """斑點：比周圍皮膚暗的小色塊 (曬斑、雀斑、痘疤、痣)。黑帽運算找局部暗點，再排除細長的(那是皺紋/毛髮)。"""
    skin = _core(pp)
    L = cv2.cvtColor(pp["face"], cv2.COLOR_BGR2LAB)[..., 0]
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    cand = (bh > 13) & skin
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    mask, n = _filter_components(cand, 8, 900, max_ecc=0.93)
    raw = 100 * mask.sum() / max(skin.sum(), 1)
    return _pack("spots", raw, n, mask, pp["regions"])


def detect_brown_spots(pp):
    """棕色斑：黑色素沉澱 (肝斑、曬斑、發炎後色素沉澱)。比周圍暗 + 偏黃棕 (Lab 的 b* 較高)。"""
    skin = _core(pp, 15)
    lab = cv2.cvtColor(pp["face"], cv2.COLOR_BGR2LAB).astype(np.float32)
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
    """紅色區：血管擴張/發炎 (泛紅、酒糟、痘痘發炎、敏感)。Lab 的 a* 明顯高於自己臉部的中位數。"""
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
    return _pack("red_areas", raw, n_pap, mask, pp["regions"], extra={"inflamed_spots": int(n_pap)})


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
    g = cv2.cvtColor(pp["face"], cv2.COLOR_BGR2GRAY).astype(np.float32)
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
        for extra in ("pigment_unevenness", "inflamed_spots"):
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
