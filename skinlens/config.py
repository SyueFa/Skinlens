"""路徑與設定。敏感設定 (資料庫連線、密碼) 從 .env 讀取，不寫在程式裡。"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "skinlens")
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"        # 原始照片 (同意用於訓練者)
PROC_DIR = DATA_DIR / "processed"        # 前處理後 (對齊/校色) 照片
HISTORY_DIR = DATA_DIR / "history"       # 會員歷史紀錄縮圖
MODEL_DIR = BASE_DIR / "models"          # 臉部偵測模型 + 訓練好的模型
# 本機測試用：有設定時，勾選「同意」的照片改存到這個資料夾 (原圖、睜眼影格、側臉、分數)，不進資料庫。
# app.py 的 HTTPS=1 (手機測試) 模式預設存到 informations/ (不進 git)
CONSENT_SAVE_DIR = os.environ.get("CONSENT_SAVE_DIR") or None

FACE_SIZE = 600       # 對齊後臉部影像寬度 (px)，所有指標都在這個尺度上計算，數值才可比較
MIN_PEERS = 10        # 同齡層至少幾筆資料才計算百分位
MAX_UPLOAD_MB = 15

for d in (UPLOAD_DIR, PROC_DIR, HISTORY_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
