"""色彩正規化：不管哪台手機、什麼燈光，盡量把照片校到同一個色彩基準，而且不洗掉膚色本身。

問題：舊做法 (Shades-of-Gray，假設整張畫面平均是灰色) 用在裁切後的臉上，畫面幾乎都是皮膚，
     會把皮膚自己的紅、黃也當成光源色偏一起校掉 (VISIA 截圖實測皮膚 a* 從 7.1 被校成 0.0)。

做法：在線性亮度空間 (光源色偏 = 各通道乘上不同係數，von Kries) 估計光源顏色：
  1. 眼白 (鞏膜)：反射率穩定、接近中性，是臉上最好的白平衡參考
     (Males et al. 2017；US 9565410 用眼白 / 牙齒 / 皮膚估計光源)。
     條件：雙眼都張開、眼白像素夠多、左右眼估計的光源一致 (FFHQ 約 79% 的照片可用)。
  2. 眼白不能用 (閉眼、太小) 時：Shades-of-Gray (線性空間)；眼白勉強可用時依可信度兩者平滑混合。
  3. 網頁拍照流程 (先睜眼看鏡頭、再閉眼拍照)：用睜眼影格的眼白估光源，套到閉眼照片上，
     並用額頭 / 臉頰膚色對齊兩者 (同一個人、同一盞燈、只隔幾秒，膚色差異 = 相機白平衡漂移)。
參考色：眼白不是純白、SoG 也有固有偏差，校色目標都改成參考族群的中位數 (scripts/fit_color_ref.py)，
     否則校成純白會整體偏藍偏冷。
試過「膚色軌跡」(只修掉偏離真實膚色帶的分量) 當備援：暖光 / 冷光的方向和膚色深淺只差約 25°，
在合成 2700K 暖光下幾乎修不掉 (ΔE 17.7 vs 不校色 19.0)，所以不用。沒有中性參考物時，
「偏黃的光」和「偏黃的皮膚」本來就分不開；要更準只能在照片裡放色卡。
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

SCLERA_MIN_PX = 25          # 每隻眼至少幾個眼白像素 (臉寬 600px)
SCLERA_MAX_LR_DEG = 6.0     # 左右眼估計的光源夾角上限 (度)
Y_BGR = np.array([0.0722, 0.7152, 0.2126])   # BGR 的亮度權重
ANCHOR_REGIONS = ("forehead", "cheek_l", "cheek_r", "nose")   # 膚色錨定用：睜眼 / 閉眼都看得到的區域

REF_FILE = Path(__file__).resolve().parent.parent.parent / "models" / "color_ref.json"
_ref = {"data": None, "mtime": None}


def to_linear(bgr):
    c = bgr.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    s = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(s * 255 + 0.5, 0, 255).astype(np.uint8)


def _angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def _reference(name):
    """族群參考色 (線性 BGR，G=1)；沒有參考檔時回傳 (1, 1, 1) = 校成純白 / 純灰。"""
    if REF_FILE.exists():
        mt = REF_FILE.stat().st_mtime
        if _ref["mtime"] != mt:
            _ref["data"], _ref["mtime"] = json.loads(REF_FILE.read_text(encoding="utf-8")), mt
        if name in _ref["data"]:
            return np.array(_ref["data"][name], float)
    return np.ones(3)


def _log_mean(illums, weights):
    """光源是乘法係數，在 log 空間加權平均 (各自先除以 G)。"""
    L = np.array([np.log(np.asarray(e, float) / e[1]) for e in illums])
    w = np.asarray(weights, float)
    return np.exp((L * w[:, None]).sum(0) / w.sum())


# ---------------------------------------------------------------- 1. 眼白
def _one_eye(lin, poly):
    m = np.zeros(lin.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0          # 避開睫毛、眼瞼邊緣
    px = lin[m]
    px = px[px.max(1) < 0.97]                                  # 過曝像素的顏色不可信
    if len(px) < SCLERA_MIN_PX * 2:
        return None, 0
    # 挑「亮、不偏色」的像素 (避開虹膜、反光、血絲)。偏色程度要在校正後才看得準，
    # 所以反覆：估光源 → 校正 → 重新挑像素，結果才不會隨照片本身的色偏改變。
    # 起點也要不受色偏影響：用眼睛區域最亮一半像素的中位數 (亮度排序幾乎不受色偏改變)，
    # 不能從「未校正的飽和度」挑，否則色偏會讓它收斂到不同的像素 (眼角粉肉、虹膜邊緣)
    y0 = px @ Y_BGR
    est = np.median(px[y0 >= np.median(y0)], axis=0)
    for _ in range(5):
        q = px / est
        y = q @ Y_BGR
        sat = (q.max(1) - q.min(1)) / (q.max(1) + 1e-6)
        lo, hi = np.percentile(y, [55, 98])
        sel = (y >= lo) & (y <= hi)
        sel &= sat <= np.percentile(sat[sel], 60) if sel.any() else sel
        if sel.sum() < SCLERA_MIN_PX:
            return None, 0
        est = np.median(px[sel], axis=0)
    return est, int(sel.sum())


def sclera_illuminant(lin, polys, use_ref=True):
    """回傳 (光源 BGR 或 None, 可信度 0~1, 說明)。
    可信度 = 眼白像素量 (每眼 75 px 以上給滿) × 左右眼一致程度 (夾角 3° 內給滿、6° 以上為 0)。
    use_ref：除以族群眼白參考色 (眼白本來就略帶黃，不是純白)。"""
    if not polys:
        return None, 0.0, "no_landmarks"
    (el, nl), (er, nr) = (_one_eye(lin, polys[k]) for k in ("eye_l", "eye_r"))
    if el is None or er is None:
        return None, 0.0, "eyes_closed_or_small"
    ang = _angle_deg(el, er)
    w = min(1.0, min(nl, nr) / (3 * SCLERA_MIN_PX)) * float(np.clip((SCLERA_MAX_LR_DEG - ang) / (SCLERA_MAX_LR_DEG / 2), 0, 1))
    est = (el + er) / 2
    return (est / _reference("sclera") if use_ref else est), w, f"px{min(nl, nr)}_lr{ang:.1f}deg"


# ---------------------------------------------------------------- 2. 備援
def shades_of_gray_illuminant(lin, p=6, use_ref=True):
    est = np.power(np.mean(np.power(lin + 1e-4, p), axis=(0, 1)), 1 / p)
    return est / _reference("sog") if use_ref else est


# ---------------------------------------------------------------- 3. 拍照流程：睜眼影格 → 閉眼照片
def skin_anchor(pp):
    """額頭、臉頰、鼻子皮膚的中位數顏色 (線性 BGR，排除最亮 / 最暗 10%)；給膚色錨定用。"""
    sel = np.zeros(pp["skin"].shape, bool)
    for k in ANCHOR_REGIONS:
        sel |= pp["regions"][k] > 0
    px = to_linear(pp["face_raw"])[sel]
    if len(px) < 500:
        return None
    y = px @ Y_BGR
    lo, hi = np.percentile(y, [10, 90])
    return np.median(px[(y >= lo) & (y <= hi)], axis=0)


def session_illuminant(frames, still):
    """睜眼影格 (多張前處理結果) 的眼白光源 → 閉眼照片的光源。回傳 (光源 BGR 或 None, 說明 dict)。
    每張影格：E_still = E_frame × (照片膚色 / 影格膚色)。同一個人、同一盞燈、只隔幾秒，
    膚色比例就是相機白平衡 / 曝光在兩者之間的漂移；最後依眼白可信度加權平均。"""
    a_still = skin_anchor(still)
    ests, ws, used = [], [], 0
    for f in frames:
        e, w, _ = sclera_illuminant(to_linear(f["face_raw"]), f["polys"])
        a_f = skin_anchor(f)
        if e is None or w <= 0 or a_f is None or a_still is None:
            continue
        ests.append(e * (a_still / a_f))
        ws.append(w)
        used += 1
    info = {"frames": len(frames), "frames_used": used}
    if not ests:
        return None, info
    if len(ests) > 1:   # 影格之間的一致程度：最大夾角 (度)
        m = _log_mean(ests, ws)
        info["spread_deg"] = round(max(_angle_deg(e / e[1], m) for e in ests), 2)
    return _log_mean(ests, ws), info


# ---------------------------------------------------------------- 套用
def apply_illuminant(lin, illum):
    """von Kries：各通道除以光源顏色；係數再乘上光源亮度，讓中性物體的亮度不變 (只改顏色、不改曝光)。"""
    illum = np.asarray(illum, np.float32)
    return lin * (float(illum @ Y_BGR) / illum)


def normalize(face_raw, polys, method="auto", illuminant=None):
    """回傳 (校色後 BGR uint8, 說明 dict)。
    illuminant：已知光源 (例如 session_illuminant 從睜眼影格估出來的)，有給就直接用。
    method: auto = 依眼白可信度把眼白與 SoG 的光源估計平滑混合 (不硬切換，邊界照片的結果才不會跳動)
            sclera = 只用眼白 (不可用就不校色) / sog = 只用 Shades-of-Gray"""
    lin = to_linear(face_raw)
    if illuminant is not None:
        illum, info = np.asarray(illuminant, float), {"method": "session"}
    else:
        sc, w, why = sclera_illuminant(lin, polys) if method in ("auto", "sclera") else (None, 0.0, "skipped")
        if method == "sog":
            illum, w = shades_of_gray_illuminant(lin), 0.0
        elif method == "sclera":
            illum, w = sc, (1.0 if sc is not None else 0.0)
        elif w >= 1.0:
            illum = sc
        elif w <= 0.0:
            illum = shades_of_gray_illuminant(lin)
        else:
            illum = _log_mean([sc, shades_of_gray_illuminant(lin)], [w, 1 - w])
        info = {"method": "sclera" if w >= 1.0 else "sog" if w <= 0.0 else "blend",
                "sclera_weight": round(float(w), 3), "sclera": why}
    if illum is None:
        info["method"] = "none"
        return face_raw.copy(), info
    illum = np.asarray(illum, float)
    info["illuminant_bgr"] = [round(float(x), 4) for x in illum / illum[1]]
    return to_srgb(apply_illuminant(lin, illum)), info
