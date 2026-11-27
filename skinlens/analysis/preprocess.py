"""
前處理：讀圖 → 臉部偵測 → 依雙眼轉正 → 裁切縮放到固定尺寸 → 色彩校正 → 品質檢查 → 切出臉部分區與皮膚遮罩。

臉部偵測兩種模式：
  1. MediaPipe FaceLandmarker (478 點，精準)：把 face_landmarker.task 放進 models/ 就會自動啟用
  2. OpenCV Haar cascade (臉框 + 眼睛)：沒有 MediaPipe 模型時的備案，分區用比例推估
"""
from __future__ import annotations
import threading
import cv2
import numpy as np
from PIL import Image, ImageOps
from ..config import MODEL_DIR, FACE_SIZE
from . import color

# ---------- MediaPipe 臉部網格的關鍵點編號 ----------
MP_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
                152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
MP_EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MP_EYE_R = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MP_BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MP_BROW_R = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
MP_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

_mp_landmarker = None
_mp_lock = threading.Lock()
_haar = None


def load_image(path_or_bytes) -> np.ndarray:
    """讀圖並依 EXIF 轉正 (手機照片常見問題)，回傳 BGR。"""
    import io
    try:
        img = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes)
    except Exception:
        raise ValueError("無法讀取圖片，請上傳 JPG / PNG 格式")
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    # 太大的圖先縮，加速偵測
    h, w = arr.shape[:2]
    s = 1600 / max(h, w)
    if s < 1:
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- 臉部偵測
def _detect_mediapipe(bgr):
    global _mp_landmarker
    task = MODEL_DIR / "face_landmarker.task"
    if not task.exists():
        return None
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with _mp_lock:   # 偵測器不能被多個執行緒同時使用 (Flask 會平行處理請求)
        if _mp_landmarker is None:
            opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(task)), num_faces=1)
            _mp_landmarker = vision.FaceLandmarker.create_from_options(opts)
        res = _mp_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
    kp = {
        "eye_l": pts[MP_EYE_L].mean(0), "eye_r": pts[MP_EYE_R].mean(0),
        "nose": pts[1], "mouth_l": pts[61], "mouth_r": pts[291], "chin": pts[152], "top": pts[10],
        "face_l": pts[234], "face_r": pts[454],
    }
    polys = {"oval": pts[MP_FACE_OVAL], "eye_l": pts[MP_EYE_L], "eye_r": pts[MP_EYE_R],
             "brow_l": pts[MP_BROW_L], "brow_r": pts[MP_BROW_R], "lips": pts[MP_LIPS]}
    return {"detector": "mediapipe", "kp": kp, "polys": polys}


def _detect_haar(bgr):
    global _haar
    if not hasattr(cv2, "CascadeClassifier"):   # OpenCV 5 已移除 Haar 偵測，沒有就跳過
        return None
    if _haar is None:
        _haar = (cv2.CascadeClassifier(str(MODEL_DIR / "haarcascade_frontalface_default.xml")),
                 cv2.CascadeClassifier(str(MODEL_DIR / "haarcascade_eye.xml")))
    face_c, eye_c = _haar
    gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi = gray[y:y + int(h * 0.55), x:x + w]
    eyes = eye_c.detectMultiScale(roi, 1.1, 6, minSize=(w // 10, w // 10), maxSize=(w // 3, w // 3))
    eyes = sorted(eyes, key=lambda e: -e[2] * e[3])[:4]
    left = [e for e in eyes if e[0] + e[2] / 2 < w / 2]
    right = [e for e in eyes if e[0] + e[2] / 2 >= w / 2]
    if left and right:
        el = np.array([x + left[0][0] + left[0][2] / 2, y + left[0][1] + left[0][3] / 2], np.float32)
        er = np.array([x + right[0][0] + right[0][2] / 2, y + right[0][1] + right[0][3] / 2], np.float32)
    else:  # 找不到眼睛就用臉框比例估
        el = np.array([x + 0.3 * w, y + 0.38 * h], np.float32)
        er = np.array([x + 0.7 * w, y + 0.38 * h], np.float32)
    ey = (el[1] + er[1]) / 2
    cx = (el[0] + er[0]) / 2
    kp = {"eye_l": el, "eye_r": er,
          "nose": np.array([cx, ey + 0.27 * h], np.float32),
          "mouth_l": np.array([cx - 0.17 * w, ey + 0.45 * h], np.float32),
          "mouth_r": np.array([cx + 0.17 * w, ey + 0.45 * h], np.float32),
          "chin": np.array([cx, y + 1.12 * h], np.float32),
          "top": np.array([cx, y - 0.08 * h], np.float32),
          "face_l": np.array([x - 0.04 * w, ey + 0.1 * h], np.float32),
          "face_r": np.array([x + 1.04 * w, ey + 0.1 * h], np.float32)}
    return {"detector": "haar", "kp": kp, "polys": None}


def detect_face(bgr):
    return _detect_mediapipe(bgr) or _detect_haar(bgr)


# ---------------------------------------------------------------- 對齊 / 校色
def _transform(det, M):
    def tf(p):
        p = np.asarray(p, np.float32).reshape(-1, 2)
        return (np.hstack([p, np.ones((len(p), 1), np.float32)]) @ M.T).astype(np.float32)
    kp = {k: tf(v)[0] for k, v in det["kp"].items()}
    polys = {k: tf(v) for k, v in det["polys"].items()} if det["polys"] else None
    return {**det, "kp": kp, "polys": polys}


def align_and_crop(bgr, det):
    """以雙眼連線轉正、裁出臉部、縮放到 FACE_SIZE 寬。"""
    kp = det["kp"]
    el, er = kp["eye_l"], kp["eye_r"]
    angle = float(np.degrees(np.arctan2(er[1] - el[1], er[0] - el[0])))
    center = tuple(((el + er) / 2).tolist())
    R = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot = cv2.warpAffine(bgr, R, (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    det = _transform(det, R)
    kp = det["kp"]
    fw = kp["face_r"][0] - kp["face_l"][0]
    x0 = kp["face_l"][0] - 0.06 * fw
    x1 = kp["face_r"][0] + 0.06 * fw
    y0 = kp["top"][1] - 0.12 * fw
    y1 = kp["chin"][1] + 0.04 * fw
    s = FACE_SIZE / (x1 - x0)
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    out_h = int((y1 - y0) * s)
    face = cv2.warpAffine(rot, M, (FACE_SIZE, out_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    det = _transform(det, M)
    warp = (np.vstack([M, [0, 0, 1]]) @ np.vstack([R, [0, 0, 1]]))[:2].astype(np.float32)   # 原圖 → 臉部座標
    return face, det, angle, fw, warp


def color_normalize(bgr, p=6):
    """舊版校色 (sRGB 上的 Shades-of-Gray)。只留給皺紋 U-Net (用這個輸入訓練的) 與皮膚遮罩用；
    指標分析改用 color.normalize (眼白 / 拍照流程，不會洗掉膚色)。"""
    f = bgr.astype(np.float32) + 1
    illum = np.power(np.mean(np.power(f, p), axis=(0, 1)), 1 / p)
    f = f * (illum.mean() / illum)
    return np.clip(f - 1, 0, 255).astype(np.uint8)


def apply_color(pp, illuminant=None):
    """(重新) 校色：更新 face、face_lin、gray_clahe、color。illuminant 給了就直接用 (拍照流程從睜眼影格估出來的)。"""
    pp["face_lin"], pp["color"] = color.normalize_linear(pp["face_raw"], pp["polys"], illuminant=illuminant)
    pp["face"] = color.to_srgb(pp["face_lin"])   # face_lin：不截斷的線性浮點影像，給色素分離用
    pp["gray_clahe"] = clahe_gray(pp["face"])
    pp.pop("_chromo_maps", None)       # 色素分離的快取跟著失效
    return pp


def clahe_gray(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


# ---------------------------------------------------------------- 分區 / 皮膚遮罩
def _ellipse(mask, c, ax, val=255):
    cv2.ellipse(mask, (int(c[0]), int(c[1])), (max(1, int(ax[0])), max(1, int(ax[1]))), 0, 0, 360, val, -1)


def _rect(shape, x0, y0, x1, y1):
    m = np.zeros(shape, np.uint8)
    cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    return m


def build_masks(face, det):
    h, w = face.shape[:2]
    kp = det["kp"]
    el, er = kp["eye_l"], kp["eye_r"]
    ipd = float(np.linalg.norm(er - el))
    ey = (el[1] + er[1]) / 2
    cx = (el[0] + er[0]) / 2
    my = (kp["mouth_l"][1] + kp["mouth_r"][1]) / 2
    shape = (h, w)

    # 臉部輪廓
    face_m = np.zeros(shape, np.uint8)
    if det["polys"]:
        cv2.fillPoly(face_m, [det["polys"]["oval"].astype(np.int32)], 255)
    else:
        top, chin = kp["top"][1], kp["chin"][1]
        _ellipse(face_m, (cx, (top + chin) / 2), ((kp["face_r"][0] - kp["face_l"][0]) / 2, (chin - top) / 2))

    # 要排除的非皮膚區：眼、眉、嘴
    excl = np.zeros(shape, np.uint8)
    if det["polys"]:
        for k in ("eye_l", "eye_r", "brow_l", "brow_r", "lips"):
            cv2.fillPoly(excl, [det["polys"][k].astype(np.int32)], 255)
        excl = cv2.dilate(excl, np.ones((int(ipd * 0.08) | 1,) * 2, np.uint8))
    else:
        for e in (el, er):
            _ellipse(excl, e, (0.32 * ipd, 0.17 * ipd))
            _ellipse(excl, (e[0], e[1] - 0.27 * ipd), (0.4 * ipd, 0.12 * ipd))
        mw = (kp["mouth_r"][0] - kp["mouth_l"][0]) / 2
        _ellipse(excl, (cx, my), (mw + 0.06 * ipd, 0.2 * ipd))
    excl |= cv2.dilate(_rect(shape, cx - 0.17 * ipd, kp["nose"][1] - 0.02 * ipd, cx + 0.17 * ipd,
                             kp["nose"][1] + 0.18 * ipd), None)  # 鼻孔

    # 自適應膚色模型：從「一定是皮膚」的種子區 (鼻樑、眼下臉頰) 取樣 Lab 顏色分佈，
    # 用 Mahalanobis 距離判斷其他像素是不是皮膚 → 不同膚色、不同燈光都適用，也能排除頭髮/背景
    lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
    seed = (_rect(shape, cx - 0.12 * ipd, ey + 0.05 * ipd, cx + 0.12 * ipd, kp["nose"][1] - 0.1 * ipd) |
            _rect(shape, el[0] - 0.2 * ipd, ey + 0.5 * ipd, el[0] + 0.15 * ipd, ey + 0.75 * ipd) |
            _rect(shape, er[0] - 0.15 * ipd, ey + 0.5 * ipd, er[0] + 0.2 * ipd, ey + 0.75 * ipd)) & face_m & ~excl
    px = lab[seed > 0]
    if len(px) > 50:
        ab = px[:, 1:]
        mu, cov = ab.mean(0), np.cov(ab.T) + np.eye(2) * 4
        d = lab[..., 1:] - mu
        md = np.einsum("...i,ij,...j->...", d, np.linalg.inv(cov), d)
        L_med = np.median(px[:, 0])
        skin_c = ((md < 9) & (lab[..., 0] > 0.55 * L_med) & (lab[..., 0] < 252)).astype(np.uint8) * 255
    else:
        ycc = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        skin_c = cv2.inRange(ycc, (40, 133, 77), (250, 180, 135))
    skin = face_m & skin_c & ~excl
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) & face_m & ~excl

    regions = {
        "forehead": _rect(shape, el[0] - 0.35 * ipd, ey - 1.0 * ipd, er[0] + 0.35 * ipd, ey - 0.45 * ipd),
        "glabella": _rect(shape, cx - 0.2 * ipd, ey - 0.45 * ipd, cx + 0.2 * ipd, ey - 0.05 * ipd),
        "crow_l": _rect(shape, el[0] - 0.68 * ipd, ey - 0.25 * ipd, el[0] - 0.3 * ipd, ey + 0.3 * ipd),
        "crow_r": _rect(shape, er[0] + 0.3 * ipd, ey - 0.25 * ipd, er[0] + 0.68 * ipd, ey + 0.3 * ipd),
        "under_eye": (_rect(shape, el[0] - 0.35 * ipd, ey + 0.18 * ipd, el[0] + 0.35 * ipd, ey + 0.45 * ipd) |
                      _rect(shape, er[0] - 0.35 * ipd, ey + 0.18 * ipd, er[0] + 0.35 * ipd, ey + 0.45 * ipd)),
        "nose": _rect(shape, cx - 0.25 * ipd, ey + 0.1 * ipd, cx + 0.25 * ipd, kp["nose"][1] + 0.05 * ipd),
        "cheek_l": _rect(shape, el[0] - 0.7 * ipd, ey + 0.48 * ipd, el[0] + 0.2 * ipd, my + 0.05 * ipd),
        "cheek_r": _rect(shape, er[0] - 0.2 * ipd, ey + 0.48 * ipd, er[0] + 0.7 * ipd, my + 0.05 * ipd),
        "nasolabial": (_rect(shape, kp["mouth_l"][0] - 0.25 * ipd, kp["nose"][1], kp["mouth_l"][0] + 0.05 * ipd, my + 0.1 * ipd) |
                       _rect(shape, kp["mouth_r"][0] - 0.05 * ipd, kp["nose"][1], kp["mouth_r"][0] + 0.25 * ipd, my + 0.1 * ipd)),
        "chin": _rect(shape, kp["mouth_l"][0], my + 0.3 * ipd, kp["mouth_r"][0], kp["chin"][1] - 0.08 * ipd),
    }
    regions = {k: (v & skin) for k, v in regions.items()}
    return skin, regions, ipd


# ---------------------------------------------------------------- 品質檢查
def quality_check(face, skin, face_px):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    sk = skin > 0
    blur = float(cv2.Laplacian(gray, cv2.CV_64F)[sk].var()) if sk.any() else 0
    bright = float(gray[sk].mean()) if sk.any() else 0
    overexp = float((gray[sk] > 245).mean()) if sk.any() else 0
    skin_ratio = float(sk.mean())
    warnings = []
    if face_px < 250: warnings.append("臉部解析度太低，請靠近一點或用更高畫質拍攝")
    if blur < 40: warnings.append("影像模糊，細紋與毛孔可能偵測不準")
    if bright < 80: warnings.append("光線太暗，建議在均勻正面光源下拍攝")
    if bright > 215 or overexp > 0.05: warnings.append("過曝或反光太多，斑點與紋理可能被洗掉")
    if skin_ratio < 0.15: warnings.append("可分析的皮膚面積太小 (可能被頭髮、眼鏡或口罩遮住)")
    return {"blur": round(blur, 1), "brightness": round(bright, 1), "overexposed": round(overexp, 3),
            "face_px": int(face_px), "skin_ratio": round(skin_ratio, 3), "warnings": warnings,
            "ok": len(warnings) == 0}


def preprocess(bgr, illuminant=None):
    """illuminant：已知光源 (拍照流程)；不給就由照片自己估 (眼白，不行再用 SoG)。"""
    det = detect_face(bgr)
    if det is None:
        raise ValueError("偵測不到臉部，請上傳正面、清楚、無遮擋的臉部照片")
    face_raw, det, roll, face_px, warp = align_and_crop(bgr, det)
    face_unet = color_normalize(face_raw)
    # 皮膚遮罩用舊版校色影像建 (自適應膚色模型，只決定「哪裡是皮膚」)：不隨校色方法改變，
    # 拍照流程重新校色時也不用重算
    skin, regions, ipd = build_masks(face_unet, det)
    q = quality_check(face_raw, skin, face_px)
    q["roll_deg"] = round(roll, 1)
    q["detector"] = det["detector"]
    pp = {"face_raw": face_raw, "face_unet": face_unet,   # face_unet：皺紋 U-Net 專用 (訓練時的輸入格式)
          "skin": skin, "regions": regions, "ipd": ipd, "kp": det["kp"], "quality": q,
          "polys": det["polys"],   # 臉部座標的五官輪廓 (眼、眉、唇、臉框)；Haar 備援時為 None
          "warp": warp}            # 2×3 仿射矩陣：原始照片座標 → 轉正裁切後的臉部座標
    return apply_color(pp, illuminant)   # face：校色後影像，所有指標都用這張
