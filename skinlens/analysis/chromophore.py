"""黑色素 / 血紅素分離 + 皮膚反光遮罩 (新增特徵用)。

方法依據：Tsumura et al., "Independent-component analysis of skin color image", JOSA A 16(9), 1999。
  皮膚顏色主要由黑色素 (棕) 與血紅素 (紅) 兩種色素決定。把 RGB 轉成光學密度 (OD = -log 反射率) 後，
  每個像素的 OD ≈ 黑色素量 × m + 血紅素量 × h + 陰影 × (1,1,1)，
  m、h 是兩種色素在相機 RGB 上的吸收方向，用 ICA 從大量皮膚像素估計 (scripts/fit_chromophore.py)。
  有了 m、h，每個像素解一個 3×3 線性方程式就能得到黑色素圖、血紅素圖，光影的影響被 (1,1,1) 項吸收。

反光：VISIA 用偏光濾鏡消除皮膚油光，一般相機沒有。這裡把「很亮又幾乎沒有顏色」的像素標成反光，
  算色素統計時排除 (簡化版的 specular highlight 處理，參考 Li et al., CVPR 2017 的問題設定)。
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np

BASIS_FILE = Path(__file__).resolve().parent.parent.parent / "models" / "chromophore.json"
_basis = {"inv": None, "mtime": None}


def optical_density(bgr):
    """BGR uint8 → RGB 光學密度 (H, W, 3)。
    JPEG 存的是 gamma 壓縮後的 sRGB，先還原成線性亮度再取 -log，模型「陰影 = 三通道等量變化」才成立。"""
    c = bgr[..., ::-1].astype(np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return -np.log(lin + 1.0 / 255.0)


def specular_mask(bgr, skin):
    """反光像素：亮度很高 (V ≥ 皮膚第 98 百分位且 > 200) 且飽和度低 (S < 45)。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    if not skin.any():
        return np.zeros(skin.shape, bool)
    thr = max(200.0, float(np.percentile(v[skin], 98)))
    m = (v >= thr) & (s < 45) & skin
    return cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def _inverse():
    """讀 models/chromophore.json 的 m、h 向量，回傳 [m h 1] 的反矩陣 (檔案更新時自動重讀)。"""
    if not BASIS_FILE.exists():
        return None
    mt = BASIS_FILE.stat().st_mtime
    if _basis["mtime"] != mt:
        b = json.loads(BASIS_FILE.read_text(encoding="utf-8"))
        A = np.column_stack([b["melanin"], b["hemoglobin"], [1.0, 1.0, 1.0]]).astype(np.float64)
        _basis["inv"], _basis["mtime"] = np.linalg.inv(A), mt
    return _basis["inv"]


def decompose(bgr):
    """回傳 (黑色素圖, 血紅素圖)；沒有色素方向檔時回傳 None。"""
    inv = _inverse()
    if inv is None:
        return None
    od = optical_density(bgr).reshape(-1, 3).astype(np.float64)
    coef = od @ inv.T
    h, w = bgr.shape[:2]
    return coef[:, 0].reshape(h, w).astype(np.float32), coef[:, 1].reshape(h, w).astype(np.float32)


def maps(pp):
    """(黑色素圖, 血紅素圖) 或 None。同一張照片只分離一次，紅色區偵測與色素特徵共用。
    pp["face"] 已在線性空間校色 (analysis/color.py)，符合這裡的線性亮度假設。"""
    if "_chromo_maps" not in pp:
        pp["_chromo_maps"] = decompose(pp["face"])
    return pp["_chromo_maps"]


def robust_stats(v):
    """中位數與 MAD 換算的標準差 (不受少數極端值影響)。"""
    med = float(np.median(v))
    return med, 1.4826 * float(np.median(np.abs(v - med))) + 1e-6


def _excess_area(img, valid, sigma=12, k=2.5):
    """比周圍高很多的區塊 (局部色素集中) 占皮膚面積 % 與塊數。門檻用自己臉上的 MAD，不受整體膚色影響。"""
    d = img - cv2.GaussianBlur(img, (0, 0), sigma)
    vals = d[valid]
    if vals.size < 100:
        return 0.0, 0
    mad = float(np.median(np.abs(vals - np.median(vals)))) + 1e-6
    hit = (d > np.median(vals) + k * 1.4826 * mad) & valid
    hit = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _ = cv2.connectedComponents(hit)
    return float(100 * hit.sum() / valid.sum()), int(n - 1)


def features(pp):
    """色素與反光特徵 (dict)。沒有色素方向檔時只回傳反光比例。"""
    face, skin = pp["face"], pp["skin"] > 0
    spec = specular_mask(face, skin)
    out = {"specular.ratio": float(100 * spec.sum() / max(skin.sum(), 1))}
    mh = maps(pp)
    valid = skin & ~spec
    if mh is None or valid.sum() < 500:
        return out
    for name, img in zip(("melanin", "hemoglobin"), mh):
        v = img[valid]
        p50, p95 = np.percentile(v, [50, 95])
        area, n = _excess_area(img, valid)
        out.update({f"chromo.{name}_mean": float(v.mean()), f"chromo.{name}_std": float(v.std()),
                    f"chromo.{name}_p95_p50": float(p95 - p50),
                    f"chromo.{name}_excess_area": area, f"chromo.{name}_excess_count": n})
        for reg, m in pp["regions"].items():
            sel = (m > 0) & valid
            if sel.sum() > 50:
                out[f"chromo.{name}.{reg}"] = float(img[sel].mean())
    return out
