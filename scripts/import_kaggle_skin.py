"""
匯入 Kaggle「Facial Skin Analysis and Type Classification」(killa92) 的人工評分，
當作斑點、棕色斑、毛孔、紅色區、紋理的訓練標註。

資料集：https://www.kaggle.com/datasets/killa92/facial-skin-analysis-and-type-classification
  4,093 張臉部照片 (640x640)，其中 200 張在兩個 Excel 檔裡有 18 項 0~5 分的人工評分。

用法：
  uv run python scripts/import_kaggle_skin.py <資料集資料夾>

欄位對應 (0~5 分 → 0~100 分)，依 skinlens/analysis/knowledge.py 的指標定義：
  Open pores                          → pores        毛孔
  Redness Severity                    → red_areas    紅色區
  uneven skin                         → texture      紋理 (資料集把它歸在 Skin Texture 類)
  dark spots / pigmentation           → brown_spots  棕色斑 (黑色素沉澱)
  max(dark spots / pigmentation, freckles, post acne marks)
                                      → spots        斑點 (定義包含雀斑、曬斑、痘疤、痣)
  ※ 兩個 Excel 檔同一欄分別叫 dark spots 與 pigmentation
額頭皺紋不匯入：只評額頭、量尺也和 FFHQ-Wrinkle 不同，混用會干擾現有的皺紋模型。

資料品質處理：
  - 同一張照片出現多次 (sha256 相同) 且評分不同 → 各欄取平均
  - Roboflow 匯出的副本 (檔名 <原始>-Copy_jpg.rf.<hash>.jpg) 記錄相同 group，
    訓練時 GroupKFold 會把同一張臉分在同一組，避免交叉驗證成績虛高
"""
import argparse, hashlib, re, sys, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import init_db, annotations, now
from skinlens.services.analyzer import extract, store_training_image

SOURCE = "kaggle_skin"
XLSX = ["skinalaysis_labeling_train1.xlsx", "skinanalysis_valid1.xlsx"]
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
COPY_SUFFIX = re.compile(r"(-Copy)?_jpg\.rf\.[0-9a-f]+\.jpg$", re.I)
CLASS_PREFIX = re.compile(r"^(oily|dry|normal|combination)_(new_)?", re.I)


def col(row, *prefixes):
    """依欄位名稱開頭找值 (不分大小寫)，找不到回傳 None。"""
    for k, v in row.items():
        if k and isinstance(v, float) and any(str(k).strip().lower().startswith(p) for p in prefixes):
            return max(0.0, min(5.0, v))
    return None


def labels_of(row):
    """一列 Excel → {我們的指標: 0~5 分}。"""
    dark = col(row, "dark spots", "pigmentation")
    out = {"pores": col(row, "open pores"), "red_areas": col(row, "redness"),
           "texture": col(row, "uneven skin"), "brown_spots": dark}
    spot_parts = [v for v in (dark, col(row, "freckles"), col(row, "post acne")) if v is not None]
    out["spots"] = max(spot_parts) if spot_parts else None
    return {k: v for k, v in out.items() if v is not None}


def original_of(name):
    """Roboflow 副本檔名 → 原始照片代號 (同一張臉的不同副本、不同膚質資料夾都會得到相同結果)。"""
    return CLASS_PREFIX.sub("", COPY_SUFFIX.sub("", name))


def read_xlsx(path):
    """讀 Excel 第一個工作表 → list[dict]。xlsx 是 zip + xml，用標準函式庫就能讀。"""
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")]
    rows = {}
    for c in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "c"):
        column, row = re.match(r"([A-Z]+)(\d+)", c.get("r")).groups()
        v = c.findtext(NS + "v")
        if v is not None:
            rows.setdefault(int(row), {})[column] = shared[int(v)] if c.get("t") == "s" else float(v)
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

    # 1) 先依照片內容 (sha256) 彙整，同一張照片多筆評分取平均
    per_image, missing = {}, 0
    for f in XLSX:
        for row in read_xlsx(base / f):
            name = str(row.get("Image_ID") or "").strip()
            if not name:
                continue
            name = name if name.lower().endswith(".jpg") else name + ".jpg"
            if name not in names:
                print(f"✗ 找不到照片 {name}"); missing += 1; continue
            data = names[name].read_bytes()
            entry = per_image.setdefault(hashlib.sha256(data).hexdigest(), {"name": name, "data": data, "labels": {}})
            for ind, v in labels_of(row).items():
                entry["labels"].setdefault(ind, []).append(v)
    dup = sum(1 for e in per_image.values() if any(len(v) > 1 for v in e["labels"].values()))

    # 2) 分析照片、寫入 MongoDB
    ok = fail = 0
    for e in per_image.values():
        try:
            pp, _, feats = extract(e["data"])
        except ValueError as err:
            print(f"✗ {e['name'][:40]}…: {err}"); fail += 1; continue
        image_id = store_training_image(e["data"], pp, feats, "standard", source=SOURCE,
                                        group=f"{SOURCE}:{original_of(e['name'])}",
                                        subject_code=f"{SOURCE}:{Path(e['name']).stem[:60]}")
        for ind, vs in e["labels"].items():
            v = sum(vs) / len(vs)
            annotations.update_one({"image_id": image_id, "indicator": ind, "annotator": SOURCE},
                                   {"$set": {"score": round(v / 5 * 100, 1), "severity": int(round(v * 4 / 5)),
                                             "n_ratings": len(vs), "updated_at": now()},
                                    "$setOnInsert": {"created_at": now()}}, upsert=True)
        ok += 1
    groups = len({original_of(e["name"]) for e in per_image.values()})
    print(f"\n完成：成功 {ok}、偵測失敗 {fail}、找不到照片 {missing}")
    print(f"不重複照片 {len(per_image)} 張 (其中 {dup} 張有多筆評分已取平均)，原始照片 {groups} 組")


if __name__ == "__main__":
    main()
