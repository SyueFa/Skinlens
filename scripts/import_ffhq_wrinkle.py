"""
匯入 FFHQ-Wrinkle 資料集 (人工皺紋遮罩) 當作「皺紋」指標的訓練標註。

資料集：https://github.com/labhai/ffhq-wrinkle-dataset
授權：CC BY-NC-SA 4.0 (禁止商業用途；照片本身各有 FFHQ 的授權)

用法：
  uv run python scripts/import_ffhq_wrinkle.py <images 資料夾> <manual_wrinkle_masks 資料夾> [--limit N]
  uv run python scripts/import_ffhq_wrinkle.py <images 資料夾> <manual_wrinkle_masks 資料夾> --relabel
      (只依遮罩重算已匯入照片的分數，不重新分析照片)

角色：
  images/                 真人照片 → 前處理 → 特徵 = 模型的輸入
  manual_wrinkle_masks/   黑底、皺紋線 = 255 的遮罩 → 只用來算答案，不會進入模型

分數換算 (兩階段，全部照片匯入後才換算)：
  1. 皺紋量 = 遮罩白線像素數 ÷ 臉部面積 (臉寬²)
     直接用整張圖當分母時，同樣的皺紋臉越大分數越高 (相關 +0.12)；除以臉部面積後為 -0.04
  2. 用與 scripts/calibrate.py 相同的族群規則換成 0~100：
     log 尺度，第 65 百分位 → 50 分、第 95 百分位 → 85 分，和其他指標的等級分布一致
     (舊版固定 0.05%→0、3.5%→100，只有 3% 的人是「優良」、41% 是「明顯 / 嚴重」，比其他指標嚴格很多)
已匯入過的照片 (同一個 subject code) 不會重新分析，可中斷後重跑。
"""
import argparse, json, math, sys
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.config import MODEL_DIR
from skinlens.db import init_db, subjects, images, annotations, now
from skinlens.services.analyzer import extract, store_training_image

SOURCE = "ffhq_wrinkle"
MID_PCT, HIGH_PCT, HIGH_SEV = 65, 95, 85
LABEL_FILE = MODEL_DIR / "wrinkle_label.json"


def wrinkle_amount(mask_path, face_px):
    """遮罩白線像素數 ÷ 臉部面積 (臉寬²)。"""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return float((m > 0).sum()) / max(float(face_px), 1.0) ** 2


def population_scores(amounts):
    """皺紋量 → 0~100 分 (log 尺度、P65→50、P95→85)，回傳 (分數 array, 換算參數)。"""
    x = np.asarray(amounts, float)
    eps = max(float(np.percentile(x, 10)), 1e-6)
    v = np.log(x + eps)
    mid, high = np.percentile(v, [MID_PCT, HIGH_PCT])
    scale = (high - mid) / math.log(HIGH_SEV / (100 - HIGH_SEV))
    return 100 / (1 + np.exp(-(v - mid) / scale)), {"eps": eps, "mid": float(mid), "scale": float(scale)}


def relabel(masks_dir):
    """依遮罩與臉部大小，重算全部 FFHQ-Wrinkle 標註的分數。"""
    rows = []
    for a in annotations.find({"annotator": SOURCE, "indicator": "wrinkles"}, {"image_id": 1}):
        im = images.find_one({"_id": a["image_id"]}, {"quality": 1, "subject_id": 1})
        code = subjects.find_one({"_id": im["subject_id"]}, {"code": 1})["code"].split(":", 1)[1]
        rows.append((a["_id"], wrinkle_amount(Path(masks_dir) / f"{code}.png", im["quality"]["face_px"])))
    if len(rows) < 100:
        sys.exit(f"只有 {len(rows)} 筆標註，族群換算至少需要 100 筆")
    scores, params = population_scores([r[1] for r in rows])
    for (aid, amount), s in zip(rows, scores):
        annotations.update_one({"_id": aid}, {"$set": {"score": round(float(s), 1), "severity": min(4, int(s // 20)),
                                                       "wrinkle_amount": amount, "updated_at": now()}})
    LABEL_FILE.write_text(json.dumps({"n": len(rows), "rule": f"log, P{MID_PCT}→50, P{HIGH_PCT}→{HIGH_SEV}",
                                      "amount": "mask_pixels / face_px^2", **params,
                                      "created_at": datetime.now().isoformat(timespec="seconds")}, indent=1),
                          encoding="utf-8")
    lv = np.bincount(np.minimum(4, (scores // 20).astype(int)), minlength=5)
    print(f"重算 {len(rows)} 筆分數；等級分布 (優良/輕微/中等/明顯/嚴重 %)：{(100 * lv / len(rows)).round(1).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("masks")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--relabel", action="store_true", help="只重算分數，不分析照片")
    a = ap.parse_args()
    init_db()
    if not a.relabel:
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
            data = ip.read_bytes()
            try:
                pp, _, feats = extract(data)
            except ValueError as e:
                print(f"✗ {mp.stem}: {e}"); fail += 1; continue
            image_id = store_training_image(data, pp, feats, "standard", subject_code=code, source=SOURCE)
            annotations.update_one({"image_id": image_id, "indicator": "wrinkles", "annotator": SOURCE},
                                   {"$setOnInsert": {"created_at": now()}}, upsert=True)
            ok += 1
            if ok % 100 == 0:
                print(f"✓ {ok} 張", flush=True)
        print(f"匯入：成功 {ok}、跳過 {skip}、失敗 {fail}")
    relabel(a.masks)


if __name__ == "__main__":
    main()
