"""從參考族群的皮膚像素估計黑色素 / 血紅素在相機 RGB 上的吸收方向 → models/chromophore.json

方法 (Tsumura et al., JOSA A 1999)：
  1. 每張照片前處理後取皮膚像素 (排除反光)，轉成光學密度 OD
  2. 去掉 (1,1,1) 方向 (光影、陰影會讓三個通道等量變化)，剩下的 2 維平面做 FastICA
  3. 兩個獨立成分的混合向量就是兩種色素的方向；依吸收特性辨認：
     血紅素在綠光吸收最強 (G 明顯高於 R、B)，黑色素由紅到藍吸收遞增 (B > R)

用法：uv run python scripts/fit_chromophore.py [--source ffhq_wrinkle] [--images 300]
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from sklearn.decomposition import FastICA
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import images
from skinlens.analysis.preprocess import load_image, preprocess
from skinlens.analysis.chromophore import optical_density, invalid_mask, face_lin, BASIS_FILE

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="ffhq_wrinkle")
ap.add_argument("--images", type=int, default=300)
ap.add_argument("--pixels", type=int, default=3000, help="每張照片取樣的皮膚像素數")
a = ap.parse_args()

POP_SIGMA = 6        # 族群基準的平滑尺度，和 indicators.detect_red_areas 的大片泛紅相同
rng = np.random.default_rng(0)
docs = list(images.find({"source": a.source}, {"raw_path": 1}).limit(a.images))
samples, od_maps = [], []
core_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))   # 和 indicators._core 相同
for d in docs:
    try:
        pp = preprocess(load_image(Path(d["raw_path"]).read_bytes()))
    except (ValueError, FileNotFoundError):
        continue
    skin = cv2.erode(pp["skin"], core_k) > 0
    lin = face_lin(pp)                # 校色後不截斷的線性浮點影像，和網站分析時用的相同
    valid = skin & ~invalid_mask(pp, skin)
    od_img = optical_density(lin)
    od = od_img[valid]
    if len(od) > a.pixels:
        od = od[rng.choice(len(od), a.pixels, replace=False)]
    samples.append(od)
    od_maps.append((od_img.astype(np.float16), valid))   # 第二步算族群基準用
X = np.concatenate(samples).astype(np.float64)
print(f"{len(samples)} 張照片、{len(X):,} 個皮膚像素")

u = np.ones(3) / np.sqrt(3)
P = np.eye(3) - np.outer(u, u)                     # 投影到與 (1,1,1) 垂直的平面
basis2 = np.linalg.svd(P)[0][:, :2]               # 平面的兩個正交基底 (3×2)
Y = (X - X.mean(0)) @ P @ basis2                  # N×2
ica = FastICA(n_components=2, whiten="unit-variance", random_state=0, max_iter=2000)
ica.fit(Y)
vecs = [basis2 @ ica.mixing_[:, i] for i in range(2)]   # 回到 RGB 空間 (平面內)
vecs = [v / np.linalg.norm(v) for v in vecs]


def hemo_score(v):       # 綠光吸收相對突出
    return v[1] - (v[0] + v[2]) / 2


def mel_score(v):        # 藍光吸收大於紅光
    return v[2] - v[0]


# 各自決定正負號，再依特性指派
vecs = [v if abs(hemo_score(v)) > abs(mel_score(v)) and hemo_score(v) > 0 or
        abs(hemo_score(v)) <= abs(mel_score(v)) and mel_score(v) > 0 else -v for v in vecs]
hi = int(np.argmax([hemo_score(v) - abs(mel_score(v)) for v in vecs]))
hemo, mel = vecs[hi], vecs[1 - hi]
if hemo_score(hemo) <= 0 or mel_score(mel) <= 0:
    sys.exit(f"無法辨認色素方向 (hemo={hemo.round(3)}, mel={mel.round(3)})，請增加照片數再試")

# 族群色素分布 (僅供參考，網站不使用)：用剛估出的方向分離每張臉，統計「平滑後」的色素濃度百分位。
# 曝光亮暗不影響色素濃度 (落在 (1,1,1) 方向)，但白平衡後殘留的光源色偏會，
# 實測暖光照片整張臉都會超過 P95，所以不拿來當偵測門檻。
inv = np.linalg.inv(np.column_stack([mel, hemo, np.ones(3)]))
pooled = {"melanin": [], "hemoglobin": []}
for od_img, valid in od_maps:
    coef = od_img.reshape(-1, 3).astype(np.float64) @ inv.T
    for i, name in enumerate(("melanin", "hemoglobin")):
        m = cv2.GaussianBlur(coef[:, i].reshape(valid.shape).astype(np.float32), (0, 0), POP_SIGMA)
        v = m[valid]
        pooled[name].append(v[rng.choice(len(v), min(len(v), a.pixels), replace=False)])
reference = {}
for name, vals in pooled.items():
    v = np.concatenate(vals)
    reference[name] = {"blur_sigma": POP_SIGMA, **{f"p{p}": round(float(np.percentile(v, p)), 5) for p in (50, 90, 95, 97)}}

out = {"source": a.source, "n_images": len(samples), "n_pixels": int(len(X)),
       "created_at": datetime.now().isoformat(timespec="seconds"),
       "melanin": [round(float(x), 5) for x in mel], "hemoglobin": [round(float(x), 5) for x in hemo],
       "reference": reference,
       "note": "RGB 光學密度空間中與 (1,1,1) 垂直平面上的方向 (Tsumura et al. 1999)；"
               "reference = 族群皮膚像素 (高斯平滑後) 的色素濃度百分位"}
BASIS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"黑色素方向 (R,G,B) = {np.round(mel, 3)}")
print(f"血紅素方向 (R,G,B) = {np.round(hemo, 3)}")
print(f"族群基準：{reference}")
print(f"→ {BASIS_FILE}")
