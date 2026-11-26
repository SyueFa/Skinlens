"""把舊版 (FastAPI + SQLite) 的 data/skin.db 搬到 MongoDB：受測者、照片 (含特徵)、人工標註。

用法：uv run python scripts/migrate_sqlite.py [data/skin.db]
重複執行沒關係：已經存在的照片 (同 sha256) 與標註會跳過。
"""
import json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pymongo.errors import DuplicateKeyError
from skinlens.db import init_db, subjects, images, annotations

path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/skin.db")
con = sqlite3.connect(path)
con.row_factory = sqlite3.Row
init_db()


def ts(v):
    if not v:
        return datetime.now(timezone.utc)
    return datetime.fromisoformat(str(v)).replace(tzinfo=timezone.utc)


def js(v):
    return json.loads(v) if isinstance(v, str) else v


subj_map = {}
for r in con.execute("select * from subjects"):
    subjects.update_one({"code": r["code"]}, {"$setOnInsert": {
        "code": r["code"], "age": r["age"], "gender": r["gender"], "skin_type": r["skin_type"],
        "fitzpatrick": r["fitzpatrick"], "consent": bool(r["consent"]), "created_at": ts(r["created_at"])}}, upsert=True)
    subj_map[r["id"]] = subjects.find_one({"code": r["code"]}, {"_id": 1})["_id"]

img_map, new_img = {}, 0
for r in con.execute("select * from images"):
    found = images.find_one({"sha256": r["sha256"]}, {"_id": 1})
    if found:
        img_map[r["id"]] = found["_id"]
        continue
    img_map[r["id"]] = images.insert_one({
        "subject_id": subj_map.get(r["subject_id"]), "sha256": r["sha256"], "raw_path": r["raw_path"],
        "processed_path": r["processed_path"], "lighting": r["lighting"] or "standard",
        "quality": js(r["quality_json"]), "features": js(r["features_json"]), "split": r["split"],
        "source": r["source"] or "upload", "created_at": ts(r["created_at"])}).inserted_id
    new_img += 1

new_ann = dup = 0
for r in con.execute("select * from annotations"):
    if r["image_id"] not in img_map:
        continue
    try:
        annotations.insert_one({"image_id": img_map[r["image_id"]], "indicator": r["indicator"],
                                "annotator": r["annotator"] or "default", "severity": r["severity"],
                                "score": r["score"], "note": r["note"], "created_at": ts(r["created_at"])})
        new_ann += 1
    except DuplicateKeyError:
        dup += 1

print(f"受測者 {len(subj_map)}、照片新增 {new_img} (共 {len(img_map)})、標註新增 {new_ann} (重複略過 {dup})")
