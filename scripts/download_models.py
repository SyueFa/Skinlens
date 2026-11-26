"""下載 MediaPipe 臉部特徵點模型 (face_landmarker.task, 約 3.7MB) 到 models/。

有它會用 478 個臉部特徵點精準切區；沒有的話自動改用內附的 OpenCV Haar 偵測 (能跑，但分區較粗)。
用法：uv run python scripts/download_models.py
"""
import sys, urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
out = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

if out.exists() and out.stat().st_size > 1_000_000:
    print(f"已經存在，不用下載：{out}")
    sys.exit(0)
out.parent.mkdir(exist_ok=True)
tmp = out.with_suffix(".part")
print(f"下載 {URL}")
urllib.request.urlretrieve(URL, tmp)
tmp.replace(out)
print(f"完成：{out} ({out.stat().st_size / 1e6:.1f} MB)")
