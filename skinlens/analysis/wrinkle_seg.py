"""AI 皺紋分割 (U-Net，用 ONNX Runtime 在 CPU 推論，不需要 PyTorch)。

模型由 scripts/train_wrinkle_seg.py 訓練、scripts/export_wrinkle_onnx.py 轉檔；
在 FFHQ-Wrinkle 測試組 100 張上 Dice 0.618 (傳統 Frangi 濾波 0.108)。
沒有模型檔時 available() 回傳 False，皺紋偵測自動退回傳統演算法。
"""
from __future__ import annotations
import json
import threading
from pathlib import Path
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
ONNX_FILE, CONF_FILE = MODEL_DIR / "wrinkle_seg.onnx", MODEL_DIR / "wrinkle_seg.json"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_state = {"session": None, "threshold": 0.5}
_lock = threading.Lock()


def available() -> bool:
    return ONNX_FILE.exists() and CONF_FILE.exists()


def _session():
    with _lock:
        if _state["session"] is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            _state["session"] = ort.InferenceSession(str(ONNX_FILE), opts, providers=["CPUExecutionProvider"])
            _state["threshold"] = float(json.loads(CONF_FILE.read_text(encoding="utf-8"))["threshold"])
    return _state["session"]


def probability(face_bgr: np.ndarray) -> np.ndarray:
    """前處理後的臉部影像 (BGR) → 每個像素是皺紋的機率 (H, W)。"""
    sess = _session()
    x = (face_bgr[..., ::-1].astype(np.float32) / 255.0 - MEAN) / STD
    h, w = x.shape[:2]
    H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    x = np.pad(x, ((0, H - h), (0, W - w), (0, 0)), mode="reflect").transpose(2, 0, 1)[None]
    logit = sess.run(None, {"image": np.ascontiguousarray(x)})[0][0, 0, :h, :w]
    return 1.0 / (1.0 + np.exp(-logit))


def predict_mask(face_bgr: np.ndarray) -> np.ndarray:
    """皺紋遮罩 (bool)，門檻值與訓練時在驗證組選出的相同。"""
    p = probability(face_bgr)
    return p > _state["threshold"]
