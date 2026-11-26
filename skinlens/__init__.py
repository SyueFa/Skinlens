"""SkinLens 膚況分析網站：Flask 應用程式工廠。"""
from datetime import timedelta
from flask import Flask, jsonify, request, render_template, g
from werkzeug.exceptions import HTTPException
from .config import SECRET_KEY, MAX_UPLOAD_MB
from .db import init_db
from .auth import bp as auth_bp, load_current_user, ensure_admin, public_user
from .routes.pages import bp as pages_bp
from .routes.api import bp as api_bp
from .routes.admin import bp as admin_bp


def create_app():
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(days=14),
        JSON_AS_ASCII=False,
    )
    app.json.ensure_ascii = False

    init_db()
    ensure_admin()

    app.before_request(load_current_user)
    for bp in (pages_bp, auth_bp, api_bp, admin_bp):
        app.register_blueprint(bp)

    @app.context_processor
    def inject_user():
        return {"current_user": public_user(g.get("user"))}

    @app.errorhandler(HTTPException)
    def http_error(e):
        msg = {404: "找不到這個頁面或資料", 413: f"檔案太大 (上限 {MAX_UPLOAD_MB}MB)",
               405: "不支援這個操作"}.get(e.code, e.description)
        if request.path.startswith("/api/"):
            return jsonify(error=msg), e.code
        return render_template("error.html", code=e.code, message=msg), e.code

    @app.errorhandler(Exception)
    def server_error(e):
        app.logger.exception("Unhandled error")
        if request.path.startswith("/api/"):
            return jsonify(error="伺服器發生錯誤，請稍後再試"), 500
        return render_template("error.html", code=500, message="伺服器發生錯誤，請稍後再試"), 500

    return app
