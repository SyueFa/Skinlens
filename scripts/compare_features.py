"""對照實驗：新增的特徵有沒有讓模型變準？

同一份標註、同樣 10 種依原始照片分組的 5 折交叉驗證，分別訓練：
  A = 舊特徵 (排除指定前綴)
  B = 全部特徵
逐次比較誤差改善 %，並報告 B 贏 A 的次數。

用法：uv run python scripts/compare_features.py [--prefix chromo. specular.] [--repeats 10]
"""
import argparse, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import INDICATORS
from skinlens.services.trainer import _dataset, _to_matrix, _new_model, MIN_SAMPLES

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", nargs="+", default=["chromo.", "specular."], help="新特徵的名稱前綴")
ap.add_argument("--repeats", type=int, default=10)
a = ap.parse_args()


def improvements(X, y, groups, base):
    out = []
    for seed in range(a.repeats):
        cv = GroupKFold(n_splits=min(5, len(set(groups))), shuffle=True, random_state=seed)
        pred = cross_val_predict(_new_model(seed), X, y, cv=cv, groups=groups)
        out.append(100 * (1 - np.mean(np.abs(pred - y)) / base))
    return np.array(out)


print(f"新特徵前綴：{a.prefix}，重複 {a.repeats} 次分組\n")
print(f"{'指標':12s} {'樣本':>5s} {'特徵A':>5s} {'特徵B':>5s}   {'A 改善':>8s}  {'B 改善':>8s}  {'差異':>7s}  B 贏的次數")
for ind in INDICATORS:
    feats, y, groups = _dataset(ind)
    if len(y) < MIN_SAMPLES:
        continue
    keys_b = sorted({k for f in feats for k in f})
    keys_a = [k for k in keys_b if not k.startswith(tuple(a.prefix))]
    if len(keys_a) == len(keys_b):
        print(f"{ind:12s} 沒有新特徵 (請先執行 reextract_features.py)"); continue
    base = float(np.mean(np.abs(y - y.mean())))
    ia = improvements(_to_matrix(feats, keys_a), y, groups, base)
    ib = improvements(_to_matrix(feats, keys_b), y, groups, base)
    print(f"{ind:12s} {len(y):5d} {len(keys_a):5d} {len(keys_b):5d}   {ia.mean():7.1f}%  {ib.mean():7.1f}%  {ib.mean() - ia.mean():+6.1f}  "
          f"{int((ib > ia).sum())}/{a.repeats}")
