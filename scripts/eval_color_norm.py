"""色彩正規化評估 (合成色偏測試)：同一張臉加上不同燈光 / 手機色偏後，校色結果能不能回到同一個顏色？

  不變性：原圖校色 vs 加色偏後校色，皮膚中位數的色度差 Δa*b* (校色只校顏色、不校曝光，所以不比亮度)
  內容穩定：同一張臉只改變裁切範圍 (像 VISIA 臉塞滿畫面)，校色結果的 ΔE。依賴整張畫面統計的方法 (SoG) 會偏掉
  保留度：校色後每個人的膚色差異還在不在 (跨人標準差；舊方法會把大家往灰色拉)。
          和原圖的相關係數只能參考：原圖本身含有拍攝時的色偏，相關高不代表比較準

色偏在線性亮度空間對各通道乘上係數 (von Kries)，模擬不同色溫 / 色調的光源與相機白平衡誤差；
係數除以最大值 (模擬相機自動曝光，亮部不過曝。不這樣做的話眼白這種最亮的參考物會被人為截掉)。
限制：手機之間的色調曲線、彩度壓縮、局部 HDR 差異沒有模擬，最終仍要用多支手機實拍驗證。

用法：uv run python scripts/eval_color_norm.py [--source ffhq_wrinkle] [--split test]
"""
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import images
from skinlens.analysis.preprocess import load_image, preprocess, color_normalize
from skinlens.analysis import color

# 色偏 (B, G, R 係數)：暖光 2700K、3500K、冷光 8000K、陰天陰影、日光燈偏綠、偏洋紅
CASTS = {"warm_2700K": (0.62, 1.0, 1.38), "warm_3500K": (0.80, 1.0, 1.18), "cool_8000K": (1.18, 1.0, 0.88),
         "shade": (1.32, 1.0, 0.80), "green": (1.0, 1.12, 1.0), "magenta": (1.0, 0.90, 1.0)}

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="ffhq_wrinkle")
ap.add_argument("--split", default="test")
a = ap.parse_args()


def methods(img, pp):
    return {"不校色": img,
            "舊 SoG (gamma)": color_normalize(img),
            "SoG (線性)": color.normalize(img, pp["polys"], "sog")[0],
            "自動 (眼白→SoG)": color.normalize(img, pp["polys"], "auto")[0]}


def skin_lab(img, skin):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[skin].astype(np.float64)
    L, A, B = np.median(lab, axis=0)
    return np.array([L * 100 / 255, A - 128, B - 128])


names = None
dE, crop_dE, base, raw_lab, used, face_method = {}, {}, {}, [], {}, []
for d in images.find({"source": a.source, "split": a.split}, {"raw_path": 1}):
    try:
        pp = preprocess(load_image(Path(d["raw_path"]).read_bytes()))
    except (ValueError, FileNotFoundError):
        continue
    skin = pp["skin"] > 0
    lin = color.to_linear(pp["face_raw"])
    m = color.normalize(pp["face_raw"], pp["polys"], "auto")[1]["method"]
    used[m] = used.get(m, 0) + 1
    face_method.append(m)
    ref = {k: skin_lab(v, skin) for k, v in methods(pp["face_raw"], pp).items()}
    names = list(ref)
    raw_lab.append(ref["不校色"])
    for k, v in ref.items():
        base.setdefault(k, []).append(v)
    for cname, g in CASTS.items():
        g = np.array(g, np.float32)
        cast = color.to_srgb(lin * (g / g.max()))     # 除以最大係數 = 相機自動曝光不讓亮部過曝
        for k, v in methods(cast, pp).items():
            dE.setdefault(k, {}).setdefault(cname, []).append(float(np.linalg.norm((skin_lab(v, skin) - ref[k])[1:])))
    # 畫面內容改變：只留臉部皮膚的外接框 (像 VISIA 那樣臉塞滿畫面)，同一張臉的校色結果會不會變
    ys, xs = np.where(skin)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    off = np.array([x0, y0], np.float32)
    pp_t = {"polys": {k: v - off for k, v in pp["polys"].items()} if pp["polys"] else None}
    for k, v in methods(pp["face_raw"][y0:y1, x0:x1], pp_t).items():
        crop_dE.setdefault(k, []).append(float(np.linalg.norm((skin_lab(v, skin[y0:y1, x0:x1]) - ref[k])[1:])))

raw_lab = np.array(raw_lab)
print(f"{a.source} / {a.split}：{len(raw_lab)} 張臉，自動模式實際使用：{used}\n")
print("不變性：加色偏後皮膚色度差 Δa*b* (越小越好；只比顏色，不比亮度)")
print(f"{'方法':16s}" + "".join(f"{c:>12s}" for c in CASTS) + f"{'平均':>8s}")
for k in names:
    row = [np.mean(dE[k][c]) for c in CASTS]
    print(f"{k:16s}" + "".join(f"{x:12.2f}" for x in row) + f"{np.mean(row):8.2f}")
fm = np.array(face_method)
for label, sel in (("  └ 眼白", fm == "sclera"), ("  └ 混合", fm == "blend"), ("  └ SoG", fm == "sog")):
    if sel.any():
        row = [np.mean(np.array(dE["自動 (眼白→SoG)"][c])[sel]) for c in CASTS]
        print(f"{label:16s}" + "".join(f"{x:12.2f}" for x in row) + f"{np.mean(row):8.2f}  ({sel.sum()} 張)")
print("\n畫面內容改變：臉塞滿畫面 (VISIA 式緊裁切) vs 原本裁切，同一張臉皮膚色度差 Δa*b* (越小越好)")
for k in names:
    v = np.array(crop_dE[k])
    print(f"{k:16s}  平均 {v.mean():5.2f}   最差 10% {np.percentile(v, 90):5.2f}")
print("\n保留度：校色後每個人的膚色差異 (和原圖的相關係數、跨人標準差)")
print(f"{'方法':16s}{'a* 相關':>9s}{'b* 相關':>9s}{'a* 標準差':>10s}{'b* 標準差':>10s}{'平均 a*':>9s}{'平均 b*':>9s}")
for k in names:
    B = np.array(base[k])
    ra = np.corrcoef(raw_lab[:, 1], B[:, 1])[0, 1]
    rb = np.corrcoef(raw_lab[:, 2], B[:, 2])[0, 1]
    print(f"{k:16s}{ra:9.2f}{rb:9.2f}{B[:, 1].std():10.2f}{B[:, 2].std():10.2f}{B[:, 1].mean():9.2f}{B[:, 2].mean():9.2f}")
