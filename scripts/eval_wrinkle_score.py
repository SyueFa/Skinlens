"""在 U-Net 完全沒看過的測試組 (100 張) 上，比較皺紋「分數」的準確度。

  A. AI 分割直接計分：U-Net 畫線 → 線條像素 ÷ 臉寬² → 和人工答案相同的換算
  B. 分數模型 (Gradient Boosting)：用舊特徵，只拿測試組以外的 900 張訓練 (公平比較)
  C. 基準線：一律猜訓練資料的平均分數
指標：平均誤差 MAE、比基準線改善 %、相關係數 r、等級完全猜中 %、差一級以內 %

用法：uv run python scripts/eval_wrinkle_score.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from skinlens.db import subjects, images, annotations
from skinlens.analysis.preprocess import load_image, preprocess
from skinlens.analysis.indicators import detect_wrinkles
from skinlens.analysis.knowledge import level_of
from skinlens.services.trainer import _to_matrix, _new_model, FEATURE_EXCLUDE

ROOT = Path(__file__).resolve().parent.parent
test_ids = set(json.loads((ROOT / "data" / "wrinkle_seg" / "split.json").read_text(encoding="utf-8"))["test"])

rows = []   # (ffhq id, 人工分數, 舊特徵, 原圖路徑)
for a in annotations.find({"annotator": "ffhq_wrinkle", "indicator": "wrinkles"}):
    im = images.find_one({"_id": a["image_id"]}, {"features": 1, "raw_path": 1, "subject_id": 1})
    pid = subjects.find_one({"_id": im["subject_id"]})["code"].split(":", 1)[1]
    rows.append((pid, a["score"], im["features"], im["raw_path"]))
train = [r for r in rows if r[0] not in test_ids]
test = [r for r in rows if r[0] in test_ids]
y_test = np.array([r[1] for r in test])
print(f"訓練 {len(train)} 張 / 測試 {len(test)} 張")

# A. AI 分割直接計分
seg = []
for r in test:
    w = detect_wrinkles(preprocess(load_image(Path(r[3]).read_bytes())))
    if w.get("method") != "unet":
        sys.exit("沒有使用 U-Net (缺少 models/wrinkle_seg.onnx 或 wrinkle_label.json)")
    seg.append(w["severity"])
seg = np.array(seg)

# B. 分數模型：只用訓練組 900 張
keys = sorted(k for k in {k for r in train for k in r[2]} if not k.startswith(FEATURE_EXCLUDE["wrinkles"]))
model = _new_model(0).fit(_to_matrix([r[2] for r in train], keys), np.array([r[1] for r in train]))
gbr = np.clip(model.predict(_to_matrix([r[2] for r in test], keys)), 0, 100)

# C. 基準線
base = np.full_like(y_test, np.mean([r[1] for r in train]))


def report(name, p):
    lv_t = np.array([level_of(v)[0] for v in y_test]); lv_p = np.array([level_of(v)[0] for v in p])
    mae, mae0 = np.mean(np.abs(p - y_test)), np.mean(np.abs(base - y_test))
    r = np.corrcoef(p, y_test)[0, 1] if p.std() > 0 else 0.0
    print(f"{name:22s} MAE {mae:5.2f}  改善 {100 * (1 - mae / mae0):5.1f}%  r {r:5.3f}  "
          f"等級猜中 {100 * np.mean(lv_t == lv_p):4.0f}%  差一級以內 {100 * np.mean(np.abs(lv_t - lv_p) <= 1):4.0f}%")
    return {"mae": round(float(mae), 2), "improvement": round(float(100 * (1 - mae / mae0)), 1), "r": round(float(r), 3),
            "level_exact": round(float(np.mean(lv_t == lv_p)), 3), "level_within1": round(float(np.mean(np.abs(lv_t - lv_p) <= 1)), 3)}


res = {"n_test": len(test), "AI 分割直接計分": report("A. AI 分割直接計分", seg),
       "分數模型 (GBR)": report("B. 分數模型 (GBR)", gbr), "基準線 (猜平均)": report("C. 基準線 (猜平均)", base)}
(ROOT / "models" / "wrinkle_score_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
