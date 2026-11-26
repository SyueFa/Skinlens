"""MongoDB 連線與資料集合 (collection)。

集合               用途                               重要欄位
users              會員                               email, name, password_hash, role (user/admin), profile
subjects           受測者 (匿名，資料集或同意者)       code, age, gender, skin_type, fitzpatrick
images             訓練用照片                          sha256, processed_path, lighting, features, split, source
annotations        人工標註 = 訓練答案                 image_id, indicator, annotator, severity (0~4), score (0~100)
analyses           每次分析結果 (會員歷史、同齡百分位)  user_id, age, lighting, overall_score, results, report
model_versions     訓練紀錄                            name, metrics, n_samples
"""
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, ASCENDING, DESCENDING
from .config import MONGO_URI, MONGO_DB

INDICATORS = ["spots", "wrinkles", "texture", "pores", "uv_spots", "brown_spots", "red_areas", "porphyrins"]
LIGHTINGS = ["standard", "cross_polarized", "uv"]
COLLATION_ZH = {"locale": "zh_Hant"}    # 繁體中文排序規則

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tz_aware=True)
db = client[MONGO_DB]
users, subjects, images = db.users, db.subjects, db.images
annotations, analyses, model_versions = db.annotations, db.analyses, db.model_versions


def now():
    return datetime.now(timezone.utc)


def oid(s):
    """字串 → ObjectId，格式錯誤回傳 None (避免使用者亂傳 id 造成 500)。"""
    if isinstance(s, ObjectId):
        return s
    try:
        return ObjectId(s)
    except (InvalidId, TypeError):
        return None


def init_db():
    """建立集合 (預設繁體中文排序) 與索引。重複執行沒關係。"""
    existing = set(db.list_collection_names())
    for name in ("users", "subjects", "images", "annotations", "analyses", "model_versions"):
        if name not in existing:
            db.create_collection(name, collation=COLLATION_ZH)
    users.create_index("email", unique=True)
    subjects.create_index("code", unique=True)
    images.create_index("sha256", unique=True)
    images.create_index([("source", ASCENDING), ("created_at", DESCENDING)])
    annotations.create_index([("image_id", ASCENDING), ("indicator", ASCENDING), ("annotator", ASCENDING)], unique=True)
    annotations.create_index("indicator")
    analyses.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    analyses.create_index([("lighting", ASCENDING), ("age", ASCENDING)])
