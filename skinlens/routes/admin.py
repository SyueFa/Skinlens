"""後台 API (限管理員)：統計、照片標註、模型訓練、匯出資料。"""
import csv
import io
import os
from flask import Blueprint, request, jsonify, send_file, abort, Response, g
from pymongo import ReturnDocument
from ..auth import admin_required
from ..db import users, subjects, images, annotations, analyses, model_versions, INDICATORS, oid, now
from ..services import trainer

bp = Blueprint("admin", __name__, url_prefix="/api/admin")


@bp.before_request
@admin_required
def _guard():
    """這個 Blueprint 底下全部 API 都要管理員權限。"""
    return None


@bp.get("/stats")
def stats():
    labeled = {r["_id"]: r["n"] for r in annotations.aggregate([
        {"$group": {"_id": {"ind": "$indicator", "img": "$image_id"}}},
        {"$group": {"_id": "$_id.ind", "n": {"$sum": 1}}}])}
    sources = {r["_id"] or "upload": r["n"] for r in images.aggregate([{"$group": {"_id": "$source", "n": {"$sum": 1}}}])}
    mv = model_versions.find_one(sort=[("created_at", -1)])
    return jsonify({
        "users": users.count_documents({}), "analyses": analyses.count_documents({}),
        "images": images.count_documents({}), "subjects": subjects.count_documents({}),
        "sources": sources, "labeled": {k: labeled.get(k, 0) for k in INDICATORS},
        "latest_model": ({"name": mv["name"], "deployed": mv.get("deployed"), "metrics": mv["metrics"],
                          "created_at": mv["created_at"].isoformat()} if mv else None),
    })


@bp.get("/images")
def list_images():
    q = {}
    if request.args.get("source"):
        q["source"] = request.args["source"]
    if request.args.get("unlabeled") == "true":
        labeled_ids = annotations.distinct("image_id")
        q["_id"] = {"$nin": labeled_ids}
    limit = min(int(request.args.get("limit", 50)), 200)
    skip = max(int(request.args.get("skip", 0)), 0)
    rows = list(images.find(q, {"features": 0}).sort("created_at", -1).skip(skip).limit(limit))
    ann = {}
    for a in annotations.find({"image_id": {"$in": [r["_id"] for r in rows]}}):
        ann.setdefault(a["image_id"], {})[a["indicator"]] = {"severity": a.get("severity"), "score": a.get("score"),
                                                             "annotator": a.get("annotator")}
    return jsonify(total=images.count_documents(q), items=[{
        "id": str(r["_id"]), "source": r.get("source"), "lighting": r.get("lighting"), "split": r.get("split"),
        "created_at": r["created_at"].isoformat(), "annotations": ann.get(r["_id"], {}),
    } for r in rows])


def _image(image_id):
    im = images.find_one({"_id": oid(image_id)}) if oid(image_id) else None
    if not im:
        abort(404)
    return im


@bp.get("/images/<image_id>/file")
def image_file(image_id):
    im = _image(image_id)
    path = im["processed_path"] if request.args.get("kind", "processed") == "processed" else im["raw_path"]
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@bp.post("/annotations")
def save_annotations():
    d = request.get_json(silent=True) or {}
    im = _image(d.get("image_id"))
    annotator = (d.get("annotator") or g.user["email"])[:40]
    saved = 0
    for it in d.get("items", []):
        ind = it.get("indicator")
        if ind not in INDICATORS:
            return jsonify(error=f"未知的指標 {ind}"), 400
        sev, score = it.get("severity"), it.get("score")
        if sev is None and score is None:
            continue
        if sev is not None and (not isinstance(sev, int) or not 0 <= sev <= 4):
            return jsonify(error="嚴重度要是 0~4 的整數"), 400
        annotations.find_one_and_update(
            {"image_id": im["_id"], "indicator": ind, "annotator": annotator},
            {"$set": {"severity": sev, "score": float(score) if score is not None else sev * 25,
                      "note": it.get("note"), "updated_at": now()},
             "$setOnInsert": {"created_at": now()}},
            upsert=True, return_document=ReturnDocument.AFTER)
        saved += 1
    return jsonify(ok=True, saved=saved)


@bp.delete("/images/<image_id>")
def delete_image(image_id):
    """受測者要求刪除資料時使用 (個資法)。"""
    im = _image(image_id)
    for p in (im.get("raw_path"), im.get("processed_path")):
        if p and os.path.exists(p):
            os.remove(p)
    annotations.delete_many({"image_id": im["_id"]})
    analyses.update_many({"image_id": im["_id"]}, {"$set": {"image_id": None}})
    images.delete_one({"_id": im["_id"]})
    return jsonify(ok=True)


@bp.post("/train")
def train():
    return jsonify(trainer.train())


@bp.get("/models")
def models():
    return jsonify([{"name": m["name"], "deployed": m.get("deployed"), "n_samples": m.get("n_samples"),
                     "metrics": m["metrics"], "created_at": m["created_at"].isoformat()}
                    for m in model_versions.find().sort("created_at", -1).limit(20)])


@bp.get("/export.csv")
def export_csv():
    """匯出 特徵 + 標註 表格，可以拿去 Jupyter 做進一步分析。"""
    ann = {}
    for a in annotations.find():
        ann.setdefault(a["image_id"], {}).setdefault(a["indicator"], []).append(a.get("score") or 0)
    rows, keys = [], set()
    for im in images.find({"features": {"$ne": None}}, {"features": 1, "split": 1, "lighting": 1, "source": 1}):
        r = {"image_id": str(im["_id"]), "source": im.get("source"), "split": im.get("split"),
             "lighting": im.get("lighting"), **im["features"]}
        for ind in INDICATORS:
            v = ann.get(im["_id"], {}).get(ind)
            r[f"label_{ind}"] = round(sum(v) / len(v), 2) if v else ""
        keys |= set(im["features"])
        rows.append(r)
    cols = ["image_id", "source", "split", "lighting"] + sorted(keys) + [f"label_{i}" for i in INDICATORS]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return Response("﻿" + buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=skinlens_dataset.csv"})
