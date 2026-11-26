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
import numpy as np
from sklearn.decomposition import FastICA
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import images
from skinlens.analysis.preprocess import load_image, preprocess
from skinlens.analysis.chromophore import optical_density, specular_mask, BASIS_FILE

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="ffhq_wrinkle")
ap.add_argument("--images", type=int, default=300)
ap.add_argument("--pixels", type=int, default=3000, help="每張照片取樣的皮膚像素數")
a = ap.parse_args()

rng = np.random.default_rng(0)
docs = list(images.find({"source": a.source}, {"raw_path": 1}).limit(a.images))
samples = []
for d in docs:
    try:
        pp = preprocess(load_image(Path(d["raw_path"]).read_bytes()))
    except (ValueError, FileNotFoundError):
        continue
    skin = pp["skin"] > 0
    valid = skin & ~specular_mask(pp["face"], skin)
    od = optical_density(pp["face"])[valid]
    if len(od) > a.pixels:
        od = od[rng.choice(len(od), a.pixels, replace=False)]
    samples.append(od)
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

out = {"source": a.source, "n_images": len(samples), "n_pixels": int(len(X)),
       "created_at": datetime.now().isoformat(timespec="seconds"),
       "melanin": [round(float(x), 5) for x in mel], "hemoglobin": [round(float(x), 5) for x in hemo],
       "note": "RGB 光學密度空間中與 (1,1,1) 垂直平面上的方向 (Tsumura et al. 1999)"}
BASIS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"黑色素方向 (R,G,B) = {np.round(mel, 3)}")
print(f"血紅素方向 (R,G,B) = {np.round(hemo, 3)}")
print(f"→ {BASIS_FILE}")
