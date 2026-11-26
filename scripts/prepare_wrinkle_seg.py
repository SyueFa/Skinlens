"""準備皺紋分割 (U-Net) 的訓練資料：全部轉到「轉正裁切後的臉部座標」。

每張 FFHQ-Wrinkle 照片：
  face.png   前處理後的臉部影像 (模型輸入，和網站分析時看到的一樣)
  mask.png   人工遮罩 (255 = 皺紋) 用同一個轉換矩陣轉到臉部座標 (答案)
  frangi.png 現有傳統演算法 (Frangi 濾波) 偵測的皺紋 (對照組)
並把照片固定分成 train 800 / val 100 / test 100 (依編號、固定亂數種子)。

用法：uv run python scripts/prepare_wrinkle_seg.py <images 資料夾> <manual_wrinkle_masks 資料夾> [--out data/wrinkle_seg]
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.analysis.preprocess import load_image, preprocess
from skinlens.analysis.indicators import detect_wrinkles

ap = argparse.ArgumentParser()
ap.add_argument("images")
ap.add_argument("masks")
ap.add_argument("--out", default="data/wrinkle_seg")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

ids = sorted(p.stem for p in Path(a.masks).glob("*.png"))
ok, failed = [], []
for i, pid in enumerate(ids):
    d = out / pid
    if (d / "frangi.png").exists():
        ok.append(pid); continue
    try:
        pp = preprocess(load_image((Path(a.images) / f"{pid}.png").read_bytes()))
    except ValueError:
        failed.append(pid); continue
    h, w = pp["face"].shape[:2]
    m = cv2.imread(str(Path(a.masks) / f"{pid}.png"), cv2.IMREAD_GRAYSCALE)
    m = cv2.warpAffine(m, pp["warp"], (w, h), flags=cv2.INTER_LINEAR, borderValue=0) > 127
    d.mkdir(exist_ok=True)
    cv2.imwrite(str(d / "face.png"), pp["face"])
    cv2.imwrite(str(d / "mask.png"), m.astype(np.uint8) * 255)
    cv2.imwrite(str(d / "frangi.png"), detect_wrinkles(pp)["mask"].astype(np.uint8) * 255)
    ok.append(pid)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(ids)}", flush=True)

rng = np.random.default_rng(42)
perm = [ok[j] for j in rng.permutation(len(ok))]
split = {"test": sorted(perm[:100]), "val": sorted(perm[100:200]), "train": sorted(perm[200:])}
(out / "split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
print(f"完成：{len(ok)} 張 (偵測失敗 {len(failed)})；train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}")
