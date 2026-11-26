"""使用者 API：膚況分析、歷史紀錄、指標知識庫。"""
import os
from flask import Blueprint, request, jsonify, g, send_file, abort
from ..auth import login_required
from ..db import analyses, LIGHTINGS, oid
from ..analysis.knowledge import INDICATORS as KB, LEVELS, GENERAL_TIPS, DISCLAIMER
from ..services import analyzer

bp = Blueprint("api", __name__, url_prefix="/api")


def _int_or_none(v, lo, hi):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None


@bp.post("/analyze")
def analyze():
    f = request.files.get("file")
    if not f:
        return jsonify(error="請選擇一張照片"), 400
    data = f.read()
    if not data:
        return jsonify(error="照片是空的，請重新選擇"), 400
    form = request.form
    prof = (g.user or {}).get("profile", {})
    lighting = form.get("lighting") if form.get("lighting") in LIGHTINGS else "standard"
    age = _int_or_none(form.get("age"), 10, 99) or prof.get("age")
    gender = form.get("gender") if form.get("gender") in ("F", "M", "other") else prof.get("gender")
    skin_type = form.get("skin_type") or prof.get("skin_type")
    consent = form.get("consent") in ("true", "1", "on")
    try:
        report = analyzer.analyze(data, lighting, age, gender, skin_type, consent, user=g.user)
    except ValueError as e:          # 讀不到圖、找不到臉等
        return jsonify(error=str(e)), 422
    return jsonify(report)


# ------------------------------------------------------------ 歷史紀錄 (只能看自己的)
def _own(aid):
    a = analyses.find_one({"_id": oid(aid), "user_id": g.user["_id"]}) if oid(aid) else None
    if not a:
        abort(404)
    return a


@bp.get("/history")
@login_required
def history():
    rows = analyses.find({"user_id": g.user["_id"]}, {"report": 0}).sort("created_at", -1).limit(200)
    return jsonify([{
        "id": str(a["_id"]), "created_at": a["created_at"].isoformat(), "lighting": a["lighting"],
        "overall_score": a["overall_score"], "summary": a["summary"],
        "scores": {k: round(100 - v["severity"], 1) for k, v in a["results"].items()},
    } for a in rows])


@bp.get("/history/<aid>")
@login_required
def history_detail(aid):
    a = _own(aid)
    return jsonify({**a["report"], "id": str(a["_id"]), "created_at": a["created_at"].isoformat(),
                    "has_image": bool(a.get("thumb_path"))})


@bp.get("/history/<aid>/image")
@login_required
def history_image(aid):
    a = _own(aid)
    if not a.get("thumb_path") or not os.path.exists(a["thumb_path"]):
        abort(404)
    return send_file(a["thumb_path"], mimetype="image/jpeg")


def _delete_analysis(a):
    if a.get("thumb_path") and os.path.exists(a["thumb_path"]):
        os.remove(a["thumb_path"])
    analyses.delete_one({"_id": a["_id"]})


@bp.delete("/history/<aid>")
@login_required
def history_delete(aid):
    _delete_analysis(_own(aid))
    return jsonify(ok=True)


@bp.delete("/history")
@login_required
def history_delete_all():
    """個資法：使用者可以一次刪除自己全部的分析紀錄與照片。"""
    rows = list(analyses.find({"user_id": g.user["_id"]}, {"thumb_path": 1}))
    for a in rows:
        _delete_analysis(a)
    return jsonify(ok=True, deleted=len(rows))


# ------------------------------------------------------------ 知識庫
@bp.get("/knowledge")
def knowledge():
    return jsonify(indicators=KB, levels=[n for _, n in LEVELS], general_tips=GENERAL_TIPS, disclaimer=DISCLAIMER)
