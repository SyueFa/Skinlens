"""
批次匯入照片 (公開資料集或自己收集的照片) 到 MongoDB 訓練資料池，並可同時匯入標註。

用法：
  uv run python scripts/import_folder.py <照片資料夾> [--labels labels.csv] [--source 資料集名稱] [--lighting standard]

labels.csv 欄位 (除了 filename 都可省略)：
  filename,age,gender,skin_type,lighting,spots,wrinkles,texture,pores,brown_spots,red_areas,uv_spots,porphyrins,annotator
  指標欄位：填整數 0~4 視為嚴重度等級；有小數點或大於 4 視為 0~100 分數。
"""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import init_db, annotations, INDICATORS, LIGHTINGS, now
from skinlens.services.analyzer import extract, store_training_image


def parse_label(v: str):
    """回傳 (severity, score)。'2' → 等級 2；'2.0'、'37' → 分數。"""
    v = v.strip()
    if v.isdigit() and int(v) <= 4:
        return int(v), int(v) * 25.0
    score = max(0.0, min(100.0, float(v)))
    return min(4, int(score // 20)), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--labels")
    ap.add_argument("--source", default="import")
    ap.add_argument("--lighting", default="standard", choices=LIGHTINGS)
    a = ap.parse_args()
    init_db()
    meta = {}
    if a.labels:
        with open(a.labels, newline="", encoding="utf-8-sig") as f:
            meta = {r["filename"]: r for r in csv.DictReader(f)}

    ok = fail = 0
    for p in sorted(Path(a.folder).iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        m = meta.get(p.name, {})
        lighting = m.get("lighting") if m.get("lighting") in LIGHTINGS else a.lighting
        data = p.read_bytes()
        try:
            pp, _, feats = extract(data, lighting)
        except ValueError as e:
            print(f"✗ {p.name}: {e}"); fail += 1; continue
        info = {k: m[k] for k in ("gender", "skin_type") if m.get(k)}
        if (m.get("age") or "").isdigit():
            info["age"] = int(m["age"])
        image_id = store_training_image(data, pp, feats, lighting, subject_code=f"{a.source}:{p.stem}",
                                        source=a.source, subject_info=info)
        annotator = m.get("annotator") or a.source
        for ind in INDICATORS:
            if (m.get(ind) or "").strip():
                sev, score = parse_label(m[ind])
                annotations.update_one({"image_id": image_id, "indicator": ind, "annotator": annotator},
                                       {"$set": {"severity": sev, "score": score, "updated_at": now()},
                                        "$setOnInsert": {"created_at": now()}}, upsert=True)
        ok += 1
        print(f"✓ {p.name} → {image_id}  quality_ok={pp['quality']['ok']}")
    print(f"\n完成：成功 {ok}、失敗 {fail}")


if __name__ == "__main__":
    main()
