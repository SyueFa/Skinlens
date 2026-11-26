"""
匯入 FFHQ-Wrinkle 資料集 (人工皺紋遮罩) 當作「皺紋」指標的訓練標註。

資料集：https://github.com/labhai/ffhq-wrinkle-dataset
授權：CC BY-NC-SA 4.0 (禁止商業用途；照片本身各有 FFHQ 的授權)

用法：
  uv run python scripts/import_ffhq_wrinkle.py <images 資料夾> <manual_wrinkle_masks 資料夾> [--limit N]

分數換算：遮罩中皺紋像素占整張圖的比例 (FFHQ 已對齊臉部，比例可比較)，
取對數後線性對應到 0~100：覆蓋率 0.05% → 0 分，3.5% → 100 分 (資料集中位數約 0.5% → 54 分)。
已匯入過的圖片 (同一個 subject code) 會跳過，可中斷後重跑。
"""
import argparse, math, sys
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import init_db, subjects, annotations, now
from skinlens.services.analyzer import extract, store_training_image

SOURCE = "ffhq_wrinkle"
LO, HI = 0.0005, 0.035


def mask_score(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    frac = max(float((m > 0).mean()), 1e-9)
    t = (math.log(frac) - math.log(LO)) / (math.log(HI) - math.log(LO))
    return frac, min(100.0, max(0.0, 100 * t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("masks")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    init_db()
    masks = sorted(Path(a.masks).glob("*.png"))[: a.limit]
    done = {s["code"] for s in subjects.find({"code": {"$regex": f"^{SOURCE}:"}}, {"code": 1})}
    ok = skip = fail = 0
    for mp in masks:
        code = f"{SOURCE}:{mp.stem}"
        ip = Path(a.images) / f"{mp.stem}.png"
        if code in done:
            skip += 1; continue
        if not ip.exists():
            print(f"✗ {mp.stem}: 找不到照片"); fail += 1; continue
        frac, score = mask_score(mp)
        data = ip.read_bytes()
        try:
            pp, _, feats = extract(data)
        except ValueError as e:
            print(f"✗ {mp.stem}: {e}"); fail += 1; continue
        image_id = store_training_image(data, pp, feats, "standard", subject_code=code, source=SOURCE)
        annotations.update_one({"image_id": image_id, "indicator": "wrinkles", "annotator": SOURCE},
                               {"$set": {"score": round(score, 1), "severity": min(4, int(score // 20)), "updated_at": now()},
                                "$setOnInsert": {"created_at": now()}}, upsert=True)
        ok += 1
        if ok % 25 == 0:
            print(f"✓ {ok} 張 (最新 {mp.stem}: 覆蓋率 {frac:.4f} → {score:.0f} 分)", flush=True)
    print(f"\n完成：成功 {ok}、跳過 {skip}、失敗 {fail}")


if __name__ == "__main__":
    main()
