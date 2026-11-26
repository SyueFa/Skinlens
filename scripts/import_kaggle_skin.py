"""
匯入 Kaggle「Facial Skin Analysis and Type Classification」(killa92) 的人工評分，
當作斑點、棕色斑、毛孔、紅色區、紋理的訓練標註。

資料集：https://www.kaggle.com/datasets/killa92/facial-skin-analysis-and-type-classification
  4,093 張臉部照片 (640x640)，其中 200 張在兩個 Excel 檔裡有 18 項 0~5 分的人工評分。

用法：
  uv run python scripts/import_kaggle_skin.py <資料集資料夾>

欄位對應 (0~5 分 → 0~100 分)：
  Open pores                  → pores        毛孔
  Redness Severity            → red_areas    紅色區
  dark spots / pigmentation   → spots、brown_spots   斑點、棕色斑
  uneven skin                 → texture      紋理
額頭皺紋不匯入：只評額頭、量尺也和 FFHQ-Wrinkle 不同，混用會干擾現有的皺紋模型。
"""
import argparse, re, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import init_db, annotations, now
from skinlens.services.analyzer import extract, store_training_image

SOURCE = "kaggle_skin"
XLSX = ["skinalaysis_labeling_train1.xlsx", "skinanalysis_valid1.xlsx"]
# 欄位名稱開頭 → 我們的指標
MAPPING = [("open pores", ["pores"]), ("redness", ["red_areas"]), ("dark spots", ["spots", "brown_spots"]),
           ("pigmentation", ["spots", "brown_spots"]), ("uneven skin", ["texture"])]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_xlsx(path):
    """讀 Excel 第一個工作表 → list[dict]。xlsx 是 zip + xml，用標準函式庫就能讀。"""
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")]
    rows = {}
    for c in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "c"):
        col, row = re.match(r"([A-Z]+)(\d+)", c.get("r")).groups()
        v = c.findtext(NS + "v")
        if v is not None:
            rows.setdefault(int(row), {})[col] = shared[int(v)] if c.get("t") == "s" else float(v)
    order = sorted(rows)
    header = rows[order[0]]
    return [{header[k]: r.get(k) for k in header} for r in (rows[i] for i in order[1:]) if r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    base = Path(a.folder)
    init_db()
    names = {p.name: p for p in base.rglob("*.jpg")}
    ok = fail = 0
    for f in XLSX:
        for row in read_xlsx(base / f):
            name = str(row.get("Image_ID") or "").strip()
            if not name:
                continue
            name = name if name.lower().endswith(".jpg") else name + ".jpg"
            if name not in names:
                print(f"✗ 找不到照片 {name}"); fail += 1; continue
            labels = {}
            for col, val in row.items():
                for prefix, inds in MAPPING:
                    if col and str(col).strip().lower().startswith(prefix) and isinstance(val, float):
                        for ind in inds:
                            labels[ind] = max(0.0, min(5.0, val))
            data = names[name].read_bytes()
            try:
                pp, _, feats = extract(data)
            except ValueError as e:
                print(f"✗ {name[:40]}…: {e}"); fail += 1; continue
            image_id = store_training_image(data, pp, feats, "standard", source=SOURCE,
                                            subject_code=f"{SOURCE}:{Path(name).stem[:60]}")
            for ind, v in labels.items():
                annotations.update_one({"image_id": image_id, "indicator": ind, "annotator": SOURCE},
                                       {"$set": {"score": round(v / 5 * 100, 1), "severity": int(round(v * 4 / 5)),
                                                 "updated_at": now()},
                                        "$setOnInsert": {"created_at": now()}}, upsert=True)
            ok += 1
    print(f"\n完成：成功 {ok}、失敗 {fail}")


if __name__ == "__main__":
    main()
