"""整條分析流程：照片 → 前處理 → 8 指標辨識 → (模型校正) → 同齡百分位 → 報告與建議 → 存入 MongoDB。"""
from __future__ import annotations
import base64
import hashlib
import uuid
import cv2
import numpy as np
from ..config import UPLOAD_DIR, PROC_DIR, HISTORY_DIR, MIN_PEERS
from ..db import subjects, images, analyses, now
from ..analysis.preprocess import load_image, preprocess
from ..analysis.indicators import analyze_all, overlay, feature_vector, COLORS
from ..analysis.knowledge import INDICATORS as KB, GENERAL_TIPS, DISCLAIMER, level_of
from . import trainer

IMAGE_KEYS = ("original", "preprocessed", "regions_preview")


def _b64(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def extract(data: bytes, lighting="standard"):
    """影像 → (前處理結果, 各指標結果, 特徵向量)。模型校正也在這裡套用。"""
    pp = preprocess(load_image(data))
    results = analyze_all(pp, lighting)
    feats = feature_vector(results, pp)
    learned = trainer.predict(feats)
    for k, r in results.items():
        if k in learned:
            r["severity_heuristic"] = r["severity"]
            r["severity"] = round(learned[k], 1)
            r["source"] = "model"
        else:
            r["source"] = "heuristic"
    return pp, results, feats


def _percentiles(results, age, lighting):
    """VISIA 的做法：跟同年齡層 (±5 歲) 的人比，回傳「你比多少 % 的人好」。"""
    if age is None:
        return {}
    rows = list(analyses.find({"lighting": lighting, "age": {"$gte": age - 5, "$lte": age + 5}}, {"results": 1}))
    out = {}
    for k, r in results.items():
        vals = [row["results"][k]["severity"] for row in rows if k in row.get("results", {})]
        if len(vals) >= MIN_PEERS:
            out[k] = round(100 * float(np.mean(np.array(vals) >= r["severity"])), 0)
    return out


def store_training_image(data, pp, feats, lighting, subject_code=None, source="upload", subject_info=None):
    """把照片存進訓練資料池 (images)。同一張照片 (sha256 相同) 只存一次。回傳 image _id。"""
    sha = hashlib.sha256(data).hexdigest()
    found = images.find_one({"sha256": sha}, {"_id": 1})
    if found:
        images.update_one({"_id": found["_id"]}, {"$set": {"features": feats}})
        return found["_id"]
    code = subject_code or "anon-" + uuid.uuid4().hex[:10]
    subjects.update_one({"code": code}, {"$setOnInsert": {"code": code, **(subject_info or {}), "consent": True,
                                                          "created_at": now()}}, upsert=True)
    subj = subjects.find_one({"code": code}, {"_id": 1})
    raw_p, proc_p = UPLOAD_DIR / f"{sha[:16]}.jpg", PROC_DIR / f"{sha[:16]}.jpg"
    raw_p.write_bytes(data)
    cv2.imwrite(str(proc_p), pp["face"])
    split = (["train"] * 8 + ["val", "test"])[int(sha[:2], 16) % 10]   # 依 hash 固定切分
    return images.insert_one({
        "subject_id": subj["_id"], "sha256": sha, "raw_path": str(raw_p), "processed_path": str(proc_p),
        "lighting": lighting, "quality": pp["quality"], "features": feats, "split": split, "source": source,
        "created_at": now()}).inserted_id


def _region_preview(pp):
    vis = pp["face"].copy()
    rng = np.random.default_rng(3)
    for m in pp["regions"].values():
        c = rng.integers(60, 255, 3)
        sel = m > 0
        vis[sel] = (vis[sel] * 0.55 + c * 0.45).astype(np.uint8)
    return vis


def build_report(pp, results, age, lighting):
    pct = _percentiles(results, age, lighting)
    indicators = []
    for k, r in results.items():
        kb = KB[k]
        lv, lv_name = level_of(r["severity"])
        worst = sorted(r["by_region"].items(), key=lambda t: -t[1])[:2]
        indicators.append({
            "key": k, "name": kb["name"], "en": kb["en"],
            "severity": r["severity"], "score": round(100 - r["severity"], 1),
            "level": lv, "level_name": lv_name, "count": r["count"], "raw": r["raw"],
            "source": r["source"], "percentile": pct.get(k),
            "worst_regions": [w[0] for w in worst if w[1] > 0],
            "what": kb["what"], "causes": kb["causes"], "advice": kb["advice"][lv],
            "doctor": kb["doctor"] if lv >= 3 else "",
            "overlay": _b64(overlay(pp["face"], r["mask"], COLORS[k])),
        })
    indicators.sort(key=lambda d: -d["severity"])
    overall = round(100 - float(np.mean([i["severity"] for i in indicators])), 1)
    concerns = [i for i in indicators if i["level"] >= 2]
    summary = (f"主要需要注意：{'、'.join(i['name'] for i in concerns[:3])}" if concerns
               else "整體膚況良好，維持防曬與保濕即可")
    return {
        "lighting": lighting, "quality": pp["quality"], "overall_score": overall, "summary": summary,
        "original": _b64(pp["face_raw"]), "preprocessed": _b64(pp["face"]),
        "regions_preview": _b64(_region_preview(pp)),
        "indicators": indicators, "general_tips": GENERAL_TIPS, "disclaimer": DISCLAIMER,
        "unavailable": ([] if lighting == "uv" else ["紫外線色斑、紫質需要 UV 光源照片，請另外上傳並選擇「UV 光源」"]),
    }


def strip_images(report):
    """存進歷史紀錄的報告不含 base64 圖片 (太大)，只留分數與建議。"""
    r = {k: v for k, v in report.items() if k not in IMAGE_KEYS}
    r["indicators"] = [{k: v for k, v in i.items() if k != "overlay"} for i in report["indicators"]]
    return r


def analyze(data: bytes, lighting="standard", age=None, gender=None, skin_type=None, consent=False, user=None):
    """網站上傳分析。會員 → 存入歷史紀錄；勾選同意 → 照片另存訓練資料池。"""
    pp, results, feats = extract(data, lighting)
    report = build_report(pp, results, age, lighting)

    image_id = None
    if consent:
        image_id = store_training_image(data, pp, feats, lighting,
                                        subject_info={"age": age, "gender": gender, "skin_type": skin_type})
    analysis_id = None
    if user or consent:
        doc = {"user_id": user["_id"] if user else None, "image_id": image_id, "lighting": lighting,
               "age": age, "gender": gender, "skin_type": skin_type,
               "overall_score": report["overall_score"], "summary": report["summary"],
               "results": {k: {"severity": r["severity"], "raw": r["raw"], "count": r["count"], "source": r["source"]}
                           for k, r in results.items()},
               "model_version": "model" if any(r["source"] == "model" for r in results.values()) else "heuristic",
               "report": strip_images(report), "created_at": now()}
        analysis_id = analyses.insert_one(doc).inserted_id
        if user:
            thumb = HISTORY_DIR / f"{analysis_id}.jpg"
            cv2.imwrite(str(thumb), pp["face"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            analyses.update_one({"_id": analysis_id}, {"$set": {"thumb_path": str(thumb)}})
    report["analysis_id"] = str(analysis_id) if analysis_id else None
    report["saved_to_history"] = bool(user)
    return report
