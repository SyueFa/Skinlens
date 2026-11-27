"""估計校色的參考色 → models/color_ref.json，給 skinlens/analysis/color.py 用。

眼白不是純白 (略帶黃、紅)，直接校成白色會讓整張照片偏藍偏冷。真實的眼白顏色沒有標準答案，
這裡用族群平均估計：參考族群 (FFHQ) 是很多台不同相機、不同燈光拍的，個別照片的白平衡誤差
平均下來會互相抵消，所以「族群中位數的眼白顏色」≈ 眼白的真實顏色。校色目標改成這個顏色。
Shades-of-Gray 同理：族群中位數的 SoG 估計值 = 它固有的偏差 (臉部畫面的平均本來就不是灰色)，校正時扣掉。

只用 train / val (test 組留給 scripts/eval_color_norm.py)。
用法：uv run python scripts/fit_color_ref.py [--source ffhq_wrinkle]
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import images
from skinlens.analysis.preprocess import load_image, preprocess
from skinlens.analysis import color

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="ffhq_wrinkle")
a = ap.parse_args()

sclera, sog, n = [], [], 0
for d in images.find({"source": a.source, "split": {"$ne": "test"}}, {"raw_path": 1}):
    try:
        pp = preprocess(load_image(Path(d["raw_path"]).read_bytes()))
    except (ValueError, FileNotFoundError):
        continue
    n += 1
    lin = color.to_linear(pp["face_raw"])
    est, w, _ = color.sclera_illuminant(lin, pp["polys"], use_ref=False)
    if est is not None and w >= 1.0:          # 只用完全可信的眼白
        sclera.append(np.log(est / est[1]))
    s = color.shades_of_gray_illuminant(lin, use_ref=False)
    sog.append(np.log(s / s[1]))

ref = {name: [round(float(x), 5) for x in np.exp(np.median(np.array(v), axis=0))] for name, v in (("sclera", sclera), ("sog", sog))}
out = {"source": a.source, "n_images": n, "n_sclera": len(sclera),
       "created_at": datetime.now().isoformat(timespec="seconds"),
       "note": "族群中位數的光源估計 (線性 BGR，G=1)。校色時把估計值除以這個參考色", **ref}
color.REF_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{n} 張照片，眼白完全可信 {len(sclera)} 張")
print(f"眼白參考色 (B, G, R) = {ref['sclera']}   SoG 參考色 = {ref['sog']}")
print(f"→ {color.REF_FILE}")
