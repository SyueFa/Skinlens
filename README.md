# SkinLens 膚況分析網站

上傳一張臉部照片 → 自動前處理 → 分析斑點、皺紋、紋理、毛孔、棕色斑、紅色區 (加 UV 照片可再測紫外線色斑、紫質) → 報告告訴你**問題在哪裡、代表什麼、怎麼改善**。會員可以保存每次結果，用趨勢圖追蹤保養前後的變化。

參考醫美診所 VISIA 膚質檢測的做法，改用一般手機照片 + 影像處理 + 機器學習校正。

| 層 | 技術 |
|---|---|
| 前端 | HTML、CSS、JavaScript、jQuery 3.7、Bootstrap 5.3、Chart.js |
| 後端 | Python 3.12、Flask 3 (Blueprint、session 登入) |
| 資料庫 | MongoDB (pymongo)，本機或 MongoDB Atlas |
| 影像分析 | OpenCV、MediaPipe Face Landmarker (478 點)、scikit-image |
| 機器學習 | scikit-learn Gradient Boosting (校正模型) |

系統架構圖與流程圖：[docs/architecture.drawio](docs/architecture.drawio) (用 draw.io 打開)。

## 快速開始

需要：[uv](https://docs.astral.sh/uv/)、MongoDB Community Server (本機 `mongodb://localhost:27017`)。

```bash
uv sync                                   # 建立 .venv 並安裝套件 (Python 3.12)
cp .env.example .env                      # 修改 SECRET_KEY、管理員帳密
uv run python scripts/download_models.py  # 下載 MediaPipe 臉部模型 (3.7MB)
uv run python app.py                      # → http://localhost:5000
```

`.env` 的 `ADMIN_EMAIL` / `ADMIN_PASSWORD` 會在啟動時自動建立管理員帳號，登入後右上角會出現「後台」。

## 頁面

| 網址 | 說明 | 權限 |
|---|---|---|
| `/` | 上傳照片、分析報告 (問題位置標示、各指標說明與建議) | 所有人 |
| `/history` | 我的紀錄：分數趨勢圖、任選兩筆前後比較、刪除 | 會員 |
| `/knowledge` | 指標百科：每個指標代表什麼、依嚴重度的建議 | 所有人 |
| `/about` | 專案介紹與系統架構 | 所有人 |
| `/account` | 個人資料 (年齡/性別/膚質)、刪除全部資料 | 會員 |
| `/admin` | 統計、照片標註 (0~4 級)、模型訓練、匯出 CSV | 管理員 |

## 專案結構

```
app.py                         Flask 入口
skinlens/
  __init__.py                  create_app()：註冊 Blueprint、錯誤處理
  config.py                    設定 (讀 .env)
  db.py                        MongoDB 連線、集合、索引 (繁體中文排序 zh_Hant)
  auth.py                      註冊 / 登入 / 登出 / 權限檢查
  routes/
    pages.py                   頁面路由 (Jinja2 樣板)
    api.py                     分析、歷史紀錄、知識庫 API
    admin.py                   後台 API (統計、標註、訓練、匯出)
  services/
    analyzer.py                分析流程：前處理 → 指標 → 模型校正 → 同齡百分位 → 報告
    trainer.py                 Gradient Boosting 訓練 / 預測
  analysis/
    preprocess.py              EXIF 轉正、臉部偵測與對齊、白平衡、CLAHE、膚色遮罩、10 分區、品質檢查
    indicators.py              8 項指標演算法
    knowledge.py               指標意義、原因、依嚴重度的建議、就醫提醒
  templates/  static/          前端 (Bootstrap + jQuery)
scripts/
  download_models.py           下載 MediaPipe 模型
  import_ffhq_wrinkle.py       匯入 FFHQ-Wrinkle 人工皺紋標註
  import_folder.py             批次匯入照片 + labels.csv
  calibrate.py                 用參考族群校準嚴重度曲線 → models/calib.json
  migrate_sqlite.py            舊版 SQLite 資料轉移到 MongoDB
  train_cnn.py                 第二階段 CNN (資料多時再用)
tests/test_app.py              端對端測試 (uv run pytest)
docs/                          架構圖 (draw.io) 與產生腳本
```

## MongoDB 集合

| 集合 | 用途 | 重要欄位 |
|---|---|---|
| `users` | 會員 | email (唯一)、name、password_hash、role (user/admin)、profile |
| `analyses` | 每次分析結果 (會員歷史、同齡百分位) | user_id、age、lighting、overall_score、results、report |
| `images` | 訓練用照片 | sha256 (唯一)、processed_path、features、split、source |
| `annotations` | 人工標註 = 訓練答案 | image_id、indicator、annotator、severity (0~4)、score (0~100) |
| `subjects` | 匿名受測者 | code、age、gender、skin_type |
| `model_versions` | 訓練紀錄 | name、deployed、metrics、n_samples |

## 分析流程

```
照片 → 前處理 (analysis/preprocess.py)
        1. EXIF 轉正、縮圖
        2. MediaPipe 偵測 478 個臉部特徵點 (沒有模型檔時改用 OpenCV Haar)
        3. 依雙眼連線轉正、裁切、縮放到寬 600px  ← 所有量測在同一尺度才能比較
        4. Shades-of-Gray 白平衡 (降低燈光色偏)、CLAHE 對比增強
        5. 自適應膚色模型 → 皮膚遮罩，排除頭髮眼眉嘴
        6. 切 10 個分區：額頭、眉間、左右眼尾、眼下、鼻、左右臉頰、法令紋、下巴
        7. 品質檢查：模糊、太暗、過曝、臉太小 → 顯示警告
     → 指標辨識 (analysis/indicators.py)：原始量測值 → 校準曲線 → 0~100 嚴重度、病灶遮罩、各分區數值
     → 模型校正 (services/trainer.py)：已上線的指標改用模型分數
     → 同齡百分位 (同年齡 ±5 歲、至少 10 筆)
     → 報告 + 建議 (analysis/knowledge.py)
```

## 準確度：怎麼校正的

1. **參考族群校準** (`scripts/calibrate.py`)：原本演算法的參數是經驗值，跟真實臉部差很多 (例如毛孔中位數 30，經驗值假設 12，幾乎人人被判「嚴重」)。改用 1,000 張 FFHQ 臉部當參考族群，在對數尺度上把第 65 百分位對到嚴重度 50、第 95 百分位對到 85。
2. **人工標註訓練** (`/admin` → 模型訓練)：用量測特徵預測人工分數的 Gradient Boosting 模型，5 折交叉驗證；只有誤差比「猜平均值」低 10% 以上的指標才上線。

目前結果 (皺紋，FFHQ-Wrinkle 1,000 張人工標註)：

| 交叉驗證誤差 (MAE) | 基準線 | 改善 | 相關係數 r |
|---|---|---|---|
| 11.17 分 | 14.58 分 | 23.4% | 0.614 |

其他指標還沒有人工標註，目前用校準後的演算法分數；在後台標註累積到每項 8 張以上就能訓練 (建議 100 張以上)。

### 重建資料 (新電腦)

```bash
# FFHQ-Wrinkle 遮罩：https://github.com/labhai/ffhq-wrinkle-dataset ；對應照片：FFHQ images1024x1024
uv run python scripts/import_ffhq_wrinkle.py <images 資料夾> <manual_wrinkle_masks 資料夾>
uv run python scripts/calibrate.py
# 再到 /admin → 模型訓練
```

資料集授權：FFHQ-Wrinkle 為 CC BY-NC-SA 4.0 (禁止商業用途)，本專案為非商業課堂作品。

## 測試

```bash
uv run pytest -q    # 使用獨立的 skinlens_test 資料庫，跑過所有頁面與 API
```

## 部署 (Render + MongoDB Atlas)

1. MongoDB Atlas 建立免費叢集，取得 `mongodb+srv://...` 連線字串
2. Render 新增 Web Service，連到這個 repo
3. Start Command：`uv run python scripts/download_models.py && uv run python app.py`
4. Environment Variables：`MONGO_URI`、`SECRET_KEY`、`ADMIN_EMAIL`、`ADMIN_PASSWORD`、`PYTHON_VERSION=3.12.6`

## 拍攝規範 (影響準確度最大的因素)

- 素顏、頭髮撥開、不要開美顏 / 濾鏡
- 正面均勻光源 (面對窗戶或環形燈)，避免頂光造成陰影被誤判成斑點 / 皺紋
- 臉部至少佔畫面寬度 1/3，對焦清楚
- 追蹤前後變化：同地點、同燈光、同距離

## 限制與隱私

- 結果是保養參考，不是醫療診斷；網頁上有免責聲明
- 一般相機沒有偏光濾鏡，表面反光會影響紅色區 / 棕色斑判斷；紫外線色斑、紫質需要 365nm UV 燈拍攝
- 沒登入且沒勾同意時，照片分析完即丟棄；會員紀錄只有本人看得到，可一鍵刪除
- 訓練資料只存匿名代碼
