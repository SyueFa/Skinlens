"""頁面路由：回傳 Jinja2 樣板，資料由前端 jQuery 透過 AJAX 向 API 取得。"""
from flask import Blueprint, render_template, g, redirect, url_for, request, send_from_directory
from ..auth import login_required, admin_required
from ..config import MODEL_DIR

bp = Blueprint("pages", __name__)


@bp.get("/")
def index():
    return render_template("index.html", page="index")


@bp.get("/favicon.ico")
def favicon():
    return redirect(url_for("static", filename="favicon.svg"), code=301)


@bp.get("/models/face_landmarker.task")
def face_model():
    """網頁拍照流程的臉部偵測模型 (和後端同一個檔案，scripts/download_models.py 下載)。只開放這一個檔案。"""
    return send_from_directory(MODEL_DIR, "face_landmarker.task", mimetype="application/octet-stream", max_age=86400)


@bp.get("/knowledge")
def knowledge():
    return render_template("knowledge.html", page="knowledge")


@bp.get("/about")
def about():
    return render_template("about.html", page="about")


@bp.get("/history")
@login_required
def history():
    return render_template("history.html", page="history")


@bp.get("/history/<aid>")
@login_required
def history_detail(aid):
    return render_template("report.html", page="history", analysis_id=aid)


@bp.get("/login")
def login():
    if g.user:
        return redirect(url_for("pages.index"))
    return render_template("login.html", page="login", mode="login", next=request.args.get("next", "/"))


@bp.get("/register")
def register():
    if g.user:
        return redirect(url_for("pages.index"))
    return render_template("login.html", page="register", mode="register", next=request.args.get("next", "/"))


@bp.get("/account")
@login_required
def account():
    return render_template("account.html", page="account")


@bp.get("/admin")
@admin_required
def admin():
    return render_template("admin.html", page="admin")
