"""用原始照片重新計算訓練資料池 (images) 的特徵。演算法或特徵有改動後執行，再重新訓練。

用法：uv run python scripts/reextract_features.py [--source kaggle_skin]
可中斷後重跑：--skip-done 會跳過已經有指定特徵的照片。
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import images
from skinlens.services.analyzer import extract

ap = argparse.ArgumentParser()
ap.add_argument("--source")
ap.add_argument("--skip-done", metavar="FEATURE", help="已經有這個特徵的照片就跳過，例如 chromo.hemoglobin_mean")
a = ap.parse_args()

q = {"source": a.source} if a.source else {}
if a.skip_done:
    # 特徵名稱本身含有「.」(例如 chromo.hemoglobin_mean)，不能寫成 features.xxx 路徑查詢，要用 $getField
    q["$expr"] = {"$eq": [{"$type": {"$getField": {"field": a.skip_done, "input": "$features"}}}, "missing"]}
total = images.count_documents(q)
ok = fail = 0
for d in images.find(q, {"raw_path": 1, "lighting": 1}).batch_size(50):
    try:
        _, _, feats = extract(Path(d["raw_path"]).read_bytes(), d.get("lighting") or "standard")
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ {d['_id']}: {e}"); fail += 1; continue
    images.update_one({"_id": d["_id"]}, {"$set": {"features": feats}})
    ok += 1
    if ok % 100 == 0:
        print(f"  {ok}/{total}", flush=True)
print(f"完成：更新 {ok}、失敗 {fail} (共 {total})")
