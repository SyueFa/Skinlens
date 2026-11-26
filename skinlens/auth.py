"""會員註冊 / 登入 / 登出 (Flask session + 雜湊密碼) 與權限檢查。"""
import re
from functools import wraps
from flask import Blueprint, request, session, jsonify, redirect, url_for, g
from pymongo.errors import DuplicateKeyError
from werkzeug.security import generate_password_hash, check_password_hash
from .db import users, now, oid
from .config import ADMIN_EMAIL, ADMIN_PASSWORD

bp = Blueprint("auth", __name__, url_prefix="/api/auth")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def public_user(u):
    if not u:
        return None
    return {"id": str(u["_id"]), "email": u["email"], "name": u.get("name") or u["email"].split("@")[0],
            "role": u.get("role", "user"), "profile": u.get("profile", {})}


def load_current_user():
    """每個請求開始時呼叫：從 session 取出目前登入的會員，放到 g.user。"""
    uid = oid(session.get("uid"))
    g.user = users.find_one({"_id": uid}) if uid else None


def _wants_json():
    return request.path.startswith("/api/")


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not g.user:
            if _wants_json():
                return jsonify(error="請先登入"), 401
            return redirect(url_for("pages.login", next=request.path))
        return fn(*a, **kw)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not g.user:
            if _wants_json():
                return jsonify(error="請先登入"), 401
            return redirect(url_for("pages.login", next=request.path))
        if g.user.get("role") != "admin":
            if _wants_json():
                return jsonify(error="需要管理員權限"), 403
            return redirect(url_for("pages.index"))
        return fn(*a, **kw)
    return wrapper


def ensure_admin():
    """依 .env 的 ADMIN_EMAIL / ADMIN_PASSWORD 建立或更新管理員帳號。"""
    if not (ADMIN_EMAIL and ADMIN_PASSWORD):
        return
    users.update_one({"email": ADMIN_EMAIL.lower()},
                     {"$set": {"password_hash": generate_password_hash(ADMIN_PASSWORD), "role": "admin"},
                      "$setOnInsert": {"name": "管理員", "created_at": now(), "profile": {}}},
                     upsert=True)


@bp.post("/register")
def register():
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    pw = d.get("password") or ""
    name = (d.get("name") or "").strip()[:30]
    if not EMAIL_RE.match(email):
        return jsonify(error="Email 格式不正確"), 400
    if len(pw) < 8:
        return jsonify(error="密碼至少 8 個字元"), 400
    try:
        r = users.insert_one({"email": email, "name": name or email.split("@")[0], "role": "user",
                              "password_hash": generate_password_hash(pw), "profile": {}, "created_at": now()})
    except DuplicateKeyError:
        return jsonify(error="這個 Email 已經註冊過了，請直接登入"), 409
    session.clear()
    session["uid"] = str(r.inserted_id)
    return jsonify(user=public_user(users.find_one({"_id": r.inserted_id})))


@bp.post("/login")
def login():
    d = request.get_json(silent=True) or {}
    u = users.find_one({"email": (d.get("email") or "").strip().lower()})
    if not u or not check_password_hash(u["password_hash"], d.get("password") or ""):
        return jsonify(error="Email 或密碼錯誤"), 401
    session.clear()
    session["uid"] = str(u["_id"])
    return jsonify(user=public_user(u))


@bp.post("/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@bp.get("/me")
def me():
    return jsonify(user=public_user(g.user))


@bp.put("/profile")
@login_required
def update_profile():
    """會員可存年齡、性別、膚質，分析時自動帶入。"""
    d = request.get_json(silent=True) or {}
    prof = {}
    if d.get("age") not in (None, ""):
        try:
            age = int(d["age"])
        except (TypeError, ValueError):
            return jsonify(error="年齡要是數字"), 400
        if not 10 <= age <= 99:
            return jsonify(error="年齡請填 10~99"), 400
        prof["age"] = age
    for k, allowed in (("gender", {"F", "M", "other"}), ("skin_type", {"oily", "dry", "combination", "normal", "sensitive"})):
        if d.get(k) in allowed:
            prof[k] = d[k]
    users.update_one({"_id": g.user["_id"]}, {"$set": {"profile": prof}})
    return jsonify(user=public_user(users.find_one({"_id": g.user["_id"]})))
