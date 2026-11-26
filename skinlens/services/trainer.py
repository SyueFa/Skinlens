"""用資料庫裡的人工標註訓練模型。

第一階段：手工特徵 + Gradient Boosting 迴歸
  - 輸入：indicators.feature_vector() 產生的量測值 (斑點面積、皺紋密度、各分區數值、膚色統計…)
  - 輸出：每個指標的 0~100 嚴重度 (對應標註者給的分數)
  - 幾十張標註就能訓練，CPU 幾秒鐘，可解釋
  - 作用：把演算法的經驗參數「校正」成跟人眼 (或醫師) 一致
  - 只有交叉驗證誤差比「猜平均值」低 10% 以上的指標才上線，避免沒學到東西的模型給使用者用
"""
from __future__ import annotations
from datetime import datetime
import threading
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from ..config import MODEL_DIR
from ..db import images, annotations, model_versions, INDICATORS, now

MODEL_PATH = MODEL_DIR / "skin_gbr.joblib"
MIN_SAMPLES = 8
DEPLOY_RATIO = 0.9
_cache = {"mtime": None, "bundle": None}
_lock = threading.Lock()


def _dataset(indicator):
    """回傳 (特徵 list, 分數 array)。同一張圖多位標註者 → 取平均 (降低個人主觀差異)。"""
    agg = {}
    for a in annotations.find({"indicator": indicator}, {"image_id": 1, "score": 1, "severity": 1}):
        score = a.get("score") if a.get("score") is not None else (a.get("severity") or 0) * 25
        agg.setdefault(a["image_id"], []).append(score)
    if not agg:
        return [], np.array([])
    feats = {im["_id"]: im["features"] for im in images.find({"_id": {"$in": list(agg)}, "features": {"$ne": None}},
                                                            {"features": 1})}
    ids = [i for i in agg if i in feats]
    return [feats[i] for i in ids], np.array([np.mean(agg[i]) for i in ids], dtype=float)


def _to_matrix(feats, keys):
    return np.array([[float(f.get(k, 0) or 0) for k in keys] for f in feats])


def train():
    if not _lock.acquire(blocking=False):
        return {"trained": False, "message": "已經有訓練在進行中，請稍候"}
    try:
        bundle = {"models": {}, "keys": {}, "metrics": {}}
        for ind in INDICATORS:
            feats, y = _dataset(ind)
            if len(y) < MIN_SAMPLES:
                bundle["metrics"][ind] = {"n": int(len(y)), "deployed": False, "status": f"樣本不足 (至少 {MIN_SAMPLES} 張)"}
                continue
            keys = sorted({k for f in feats for k in f})
            X = _to_matrix(feats, keys)
            model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                              subsample=0.8, random_state=0)
            pred = cross_val_predict(model, X, y, cv=KFold(min(5, len(y)), shuffle=True, random_state=0))
            mae = float(np.mean(np.abs(pred - y)))
            baseline = float(np.mean(np.abs(y - y.mean())))
            corr = float(np.corrcoef(pred, y)[0, 1]) if y.std() > 0 else 0.0
            model.fit(X, y)
            use = mae < baseline * DEPLOY_RATIO
            top = sorted(zip(keys, model.feature_importances_), key=lambda t: -t[1])[:5]
            if use:
                bundle["models"][ind] = model
                bundle["keys"][ind] = keys
            bundle["metrics"][ind] = {
                "n": int(len(y)), "deployed": use, "cv_mae": round(mae, 2), "baseline_mae": round(baseline, 2),
                "improvement": round(100 * (1 - mae / baseline), 1) if baseline else 0.0,
                "pearson_r": round(corr, 3), "top_features": [[a, round(float(b), 3)] for a, b in top]}
        name = "gbr-" + datetime.now().strftime("%Y%m%d-%H%M%S")
        trained = bool(bundle["models"])
        if trained:
            joblib.dump(bundle, MODEL_PATH)
            _cache["mtime"] = None
        model_versions.insert_one({"name": name, "kind": "sklearn", "deployed": trained, "metrics": bundle["metrics"],
                                   "n_samples": max((m["n"] for m in bundle["metrics"].values()), default=0),
                                   "created_at": now()})
        msg = "訓練完成" if trained else "沒有任何指標的模型優於基準線，暫不上線 (需要更多或更一致的標註)"
        return {"trained": trained, "version": name, "metrics": bundle["metrics"], "message": msg}
    finally:
        _lock.release()


def load():
    if not MODEL_PATH.exists():
        return None
    mt = MODEL_PATH.stat().st_mtime
    if _cache["mtime"] != mt:
        _cache["bundle"], _cache["mtime"] = joblib.load(MODEL_PATH), mt
    return _cache["bundle"]


def predict(features: dict) -> dict:
    """回傳 {indicator: severity}，只包含已上線的指標。"""
    b = load()
    if not b:
        return {}
    return {ind: float(np.clip(m.predict(_to_matrix([features], b["keys"][ind]))[0], 0, 100))
            for ind, m in b["models"].items()}
