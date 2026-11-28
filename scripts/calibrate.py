"""用資料庫裡的參考族群 (預設 FFHQ-Wrinkle 1,000 張臉) 校準演算法的嚴重度曲線。

原本 indicators.CALIB 是經驗值，跟真實臉部分布差很多 (例如毛孔中位數 30，經驗值卻假設 12，
幾乎每個人都被判成「嚴重」)。這裡改成跟參考族群比較 (VISIA 的概念)：
  - 第 65 百分位的人 → 嚴重度 50
  - 第 95 百分位的人 → 嚴重度 85
  - 在 log(量測值 + eps) 上計算 (分布右偏)
結果寫到 models/calib.json，indicators.py 啟動時自動讀取。

用法：uv run python scripts/calibrate.py [--source ffhq_wrinkle]
"""
import argparse, json, math, sys
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import images
from skinlens.config import MODEL_DIR

VISIBLE = ["spots", "wrinkles", "texture", "pores", "brown_spots", "red_areas"]
MID_PCT, HIGH_PCT, HIGH_SEV = 65, 95, 85
SHARP_ONLY = {"pores": 50}     # 只用清晰度前 N% 的參考照片校準的指標

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="ffhq_wrinkle")
a = ap.parse_args()

feats = [im["features"] for im in images.find({"source": a.source, "features": {"$ne": None}}, {"features": 1})]
if len(feats) < 100:
    sys.exit(f"參考族群只有 {len(feats)} 張，至少要 100 張")

calib, summary = {}, {}
k_high = math.log(HIGH_SEV / (100 - HIGH_SEV))
blur_all = np.array([f.get("quality.blur", 0) for f in feats], float)
for ind in VISIBLE:
    ref = feats
    if ind in SHARP_ONLY:
        # 毛孔要夠清楚才看得到：模糊的參考照片幾乎量不到毛孔，會把族群基準拉低，
        # 手機近拍的人全被判成「嚴重」。只用清晰度前 SHARP_ONLY% 的參考照片
        th = np.percentile(blur_all, 100 - SHARP_ONLY[ind])
        ref = [f for f in feats if f.get("quality.blur", 0) >= th]
    raw = np.array([f[f"{ind}.raw"] for f in ref if f.get(f"{ind}.raw") is not None], float)
    # 量測值多半右偏 (多數人很少、少數人很多)，在對數尺度上校準，皮膚乾淨的人才拿得到「優良」
    eps = max(float(np.percentile(raw, 10)), 0.05 * float(np.median(raw)), 1e-3)
    v = np.log(raw + eps)
    mid, high = np.percentile(v, [MID_PCT, HIGH_PCT])
    scale = max((high - mid) / k_high, 1e-3)
    calib[ind] = [round(float(mid), 4), round(float(scale), 4), round(eps, 4)]
    sev = 100 / (1 + np.exp(-(v - mid) / scale))
    summary[ind] = {"median_raw": round(float(np.median(raw)), 3), "n": int(len(raw)),
                    "levels_pct": [round(float(np.mean((sev >= lo) & (sev < hi))) * 100, 1)
                                   for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]]}

out = MODEL_DIR / "calib.json"
out.write_text(json.dumps({"source": a.source, "n": len(feats), "created_at": datetime.now().isoformat(timespec="seconds"),
                           "rule": f"P{MID_PCT}→50, P{HIGH_PCT}→{HIGH_SEV}", "calib": calib, "summary": summary},
                          ensure_ascii=False, indent=1), encoding="utf-8")
print(f"參考族群 {a.source}：{len(feats)} 張 → {out}")
print("指標          mid(log)  scale   優良/輕微/中等/明顯/嚴重 (%)")
for ind in VISIBLE:
    print(f"{ind:12s} {calib[ind][0]:8.3f} {calib[ind][1]:8.3f}   {summary[ind]['levels_pct']}")
