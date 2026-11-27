"""下載 MediaPipe 臉部模型與網頁版偵測程式。

1. face_landmarker.task (約 3.7MB) → models/
   網站必須有這個模型才能偵測臉部 (OpenCV 5 已移除 Haar 偵測，只有 OpenCV 4 環境才有備援)。
   網頁拍照流程也用同一個檔案 (由 /models/face_landmarker.task 提供)。
2. @mediapipe/tasks-vision 網頁版 (約 15MB，Apache-2.0) → skinlens/static/vendor/mediapipe/
   網頁拍照流程用來即時偵測頭部角度，引導使用者正對鏡頭才拍 (capture.js)。
   放在自己的伺服器，不從外部 CDN 載入。沒下載的話拍照流程退回只按秒數倒數。

用法：uv run python scripts/download_models.py
"""
import io, sys, tarfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
VISION_VER = "1.2.3"
VISION_URL = f"https://registry.npmjs.org/@mediapipe/tasks-vision/-/tasks-vision-{VISION_VER}.tgz"
# 只取用得到的檔案 (tarball 裡的路徑 → 存檔名稱)
VISION_FILES = {"package/vision_bundle.mjs": "vision_bundle.mjs",
                "package/wasm/vision_wasm_internal.js": "wasm/vision_wasm_internal.js",
                "package/wasm/vision_wasm_internal.wasm": "wasm/vision_wasm_internal.wasm"}


def get_task():
    out = ROOT / "models" / "face_landmarker.task"
    if out.exists() and out.stat().st_size > 1_000_000:
        print(f"已經存在，不用下載：{out}")
        return
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".part")
    print(f"下載 {TASK_URL}")
    urllib.request.urlretrieve(TASK_URL, tmp)
    tmp.replace(out)
    print(f"完成：{out} ({out.stat().st_size / 1e6:.1f} MB)")


def get_vision():
    dst = ROOT / "skinlens" / "static" / "vendor" / "mediapipe"
    stamp = dst / "VERSION"
    if stamp.exists() and stamp.read_text().strip() == VISION_VER and all((dst / f).exists() for f in VISION_FILES.values()):
        print(f"已經存在，不用下載：{dst} ({VISION_VER})")
        return
    print(f"下載 {VISION_URL}")
    data = urllib.request.urlopen(VISION_URL).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for name, rel in VISION_FILES.items():
            f = tar.extractfile(tar.getmember(name))   # 只讀指定的檔案，不整包解開
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(f.read())
    stamp.write_text(VISION_VER)
    print(f"完成：{dst}")


if __name__ == "__main__":
    get_task()
    try:
        get_vision()
    except Exception as e:   # 網頁版只是加分功能，失敗不影響網站啟動
        print(f"網頁版偵測程式下載失敗 (拍照流程會退回只按秒數倒數)：{e}", file=sys.stderr)
