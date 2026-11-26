"""把 U-Net 皺紋分割權重轉成 ONNX，網站用 onnxruntime 在 CPU 推論 (不需要安裝 PyTorch)。

用法：uv run --extra train python scripts/export_wrinkle_onnx.py
輸入 models/wrinkle_seg.pt → 輸出 models/wrinkle_seg.onnx、models/wrinkle_seg.json (門檻值等設定)
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
ckpt = torch.load(ROOT / "models" / "wrinkle_seg.pt", map_location="cpu")
model = smp.Unet(ckpt["encoder"], encoder_weights=None, classes=1)
model.load_state_dict(ckpt["state_dict"])
model.eval()
x = torch.randn(1, 3, 704, 608)
out = ROOT / "models" / "wrinkle_seg.onnx"
torch.onnx.export(model, x, str(out), input_names=["image"], output_names=["logit"], opset_version=17,
                  dynamic_axes={"image": {2: "h", 3: "w"}, "logit": {2: "h", 3: "w"}}, dynamo=False)
with torch.no_grad():
    ref = model(x).numpy()
import onnxruntime as ort
got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
diff = float(np.abs(ref - got).max())
(ROOT / "models" / "wrinkle_seg.json").write_text(json.dumps(
    {"encoder": ckpt["encoder"], "threshold": ckpt["threshold"], "input": "RGB, ImageNet 標準化, 長寬補到 32 的倍數",
     "onnx_max_abs_diff_vs_torch": diff}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"→ {out} ({out.stat().st_size / 1e6:.1f} MB)，與 PyTorch 輸出最大差異 {diff:.2e}")
