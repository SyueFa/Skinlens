"""用資料庫裡的人工標註訓練模型。

第一階段：手工特徵 + Gradient Boosting 迴歸
  - 輸入：indicators.feature_vector() 產生的量測值 (斑點面積、皺紋密度、各分區數值、膚色統計…)
  - 輸出：每個指標的 0~100 嚴重度 (對應標註者給的分數)
  - 幾十張標註就能訓練，CPU 幾秒鐘，可解釋
  - 作用：把演算法的經驗參數「校正」成跟人眼 (或醫師) 一致
  - 依原始照片分組 (GroupKFold)、重複 10 次不同分組的交叉驗證，平均誤差比「猜平均值」低 10% 以上才上線，
    避免同一張臉的副本洩漏、或一次分組運氣好而讓沒學到東西的模型上線
"""
from __future__ import annotations
from datetime import datetime
import threading
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from ..config import MODEL_DIR
from ..db import images, annotations, model_versions, INDICATORS, now

MODEL_PATH = MODEL_DIR / "skin_gbr.joblib"
MIN_SAMPLES = 8
DEPLOY_RATIO = 0.9
# 看「形狀」的指標不用色素 / 反光特徵：對照實驗 (scripts/compare_features.py) 顯示加入後皺紋 -0.5、毛孔 -1.9 個百分點，
# 顏色類指標則有幫助 (紅色區 +8.1、紋理 +2.6、棕色斑 +2.2、斑點 +0.8)
FEATURE_EXCLUDE = {"wrinkles": ("chromo.", "specular."), "pores": ("chromo.", "specular.")}
CV_REPEATS = 10       # 重複幾次不同的隨機分組；樣本少時只跑一次容易因為運氣好而誤上線
_cache = {"mtime": None, "bundle": None}
_lock = threading.Lock()


def _dataset(indicator):
    """回傳 (特徵 list, 分數 array, 分組 array)。同一張圖多位標註者 → 取平均 (降低個人主觀差異)。
    分組：同一張原始照片的不同副本 (images.group 相同) 屬於同一組，交叉驗證時不會被拆到訓練 / 驗證兩邊。"""
    agg = {}
    for a in annotations.find({"indicator": indicator}, {"image_id": 1, "score": 1, "severity": 1}):
        score = a.get("score") if a.get("score") is not None else (a.get("severity") or 0) * 25
        agg.setdefault(a["image_id"], []).append(score)
    if not agg:
        return [], np.array([]), np.array([])
    docs = {im["_id"]: im for im in images.find({"_id": {"$in": list(agg)}, "features": {"$ne": None}},
                                                {"features": 1, "group": 1})}
    ids = [i for i in agg if i in docs]
    groups = np.array([docs[i].get("group") or str(i) for i in ids])
    return [docs[i]["features"] for i in ids], np.array([np.mean(agg[i]) for i in ids], dtype=float), groups


def _to_matrix(feats, keys):
    return np.array([[float(f.get(k, 0) or 0) for k in keys] for f in feats])


def _new_model(seed):
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)


def train():
    if not _lock.acquire(blocking=False):
        return {"trained": False, "message": "已經有訓練在進行中，請稍候"}
    try:
        bundle = {"models": {}, "keys": {}, "metrics": {}}
        for ind in INDICATORS:
            feats, y, groups = _dataset(ind)
            if len(y) < MIN_SAMPLES:
                bundle["metrics"][ind] = {"n": int(len(y)), "deployed": False, "status": f"樣本不足 (至少 {MIN_SAMPLES} 張)"}
                continue
            keys = sorted(k for k in {k for f in feats for k in f} if not k.startswith(FEATURE_EXCLUDE.get(ind, ())))
            X = _to_matrix(feats, keys)
            baseline = float(np.mean(np.abs(y - y.mean())))
            maes, corrs = [], []
            for seed in range(CV_REPEATS):      # 重複分組交叉驗證，取平均
                cv = GroupKFold(n_splits=min(5, len(set(groups))), shuffle=True, random_state=seed)
                pred = cross_val_predict(_new_model(seed), X, y, cv=cv, groups=groups)
                maes.append(float(np.mean(np.abs(pred - y))))
                corrs.append(float(np.corrcoef(pred, y)[0, 1]) if y.std() > 0 else 0.0)
            mae, corr = float(np.mean(maes)), float(np.mean(corrs))
            model = _new_model(0)
            model.fit(X, y)
            use = mae < baseline * DEPLOY_RATIO
            top = sorted(zip(keys, model.feature_importances_), key=lambda t: -t[1])[:5]
            if use:
                bundle["models"][ind] = model
                bundle["keys"][ind] = keys
            bundle["metrics"][ind] = {
                "n": int(len(y)), "groups": int(len(set(groups))), "deployed": use, "cv_mae": round(mae, 2), "baseline_mae": round(baseline, 2),
                "improvement": round(100 * (1 - mae / baseline), 1) if baseline else 0.0,
                "improvement_range": ([round(100 * (1 - max(maes) / baseline), 1), round(100 * (1 - min(maes) / baseline), 1)]
                                      if baseline else [0.0, 0.0]),
                "cv_repeats": CV_REPEATS,
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
