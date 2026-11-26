"""端對端測試：用 Flask test client 跑過所有頁面與 API。使用獨立的測試資料庫，不影響正式資料。
執行：uv run python -m pytest -q tests  (或 uv run python tests/test_app.py)"""
import os, sys, glob, io
from pathlib import Path
from PIL import Image as PILImage
os.environ["MONGO_DB"] = "skinlens_test"
os.environ["ADMIN_EMAIL"] = "admin@test.local"
os.environ["ADMIN_PASSWORD"] = "admin-pass-123"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skinlens.db import client
client.drop_database("skinlens_test")
from skinlens import create_app

FACE = sorted(glob.glob(r"D:/code/visia/data/processed/*.jpg"))[0]
app = create_app()
app.testing = True


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


def run():
    c = app.test_client()
    print("[頁面 - 未登入]")
    for p in ["/", "/knowledge", "/about", "/login", "/register"]:
        check(c.get(p).status_code == 200, f"GET {p} 200")
    for p in ["/history", "/account", "/admin"]:
        r = c.get(p)
        check(r.status_code == 302 and "/login" in r.headers["Location"], f"GET {p} 未登入 → 導向登入")
    check(c.get("/nope").status_code == 404, "不存在頁面 404")
    check(c.get("/api/history").status_code == 401, "API 未登入 401")
    check(c.get("/api/admin/stats").status_code == 401, "後台 API 未登入 401")
    check(c.get("/api/knowledge").get_json()["indicators"]["wrinkles"]["name"] == "皺紋", "知識庫中文正常")

    print("[分析 - 未登入]")
    check(c.post("/api/analyze", data={}).status_code == 400, "沒附檔案 400")
    r = c.post("/api/analyze", data={"file": (io.BytesIO(b"not an image"), "x.jpg")})
    check(r.status_code == 422, f"非圖片檔 422 ({r.status_code} {r.get_json()})")
    blank = io.BytesIO()
    PILImage.new("RGB", (640, 480), (180, 170, 160)).save(blank, "JPEG"); blank.seek(0)
    r = c.post("/api/analyze", data={"file": (blank, "blank.jpg")})
    check(r.status_code == 422 and "偵測不到臉部" in r.get_json()["error"], f"沒有臉的圖片 422 + 提示 ({r.status_code} {r.get_json()})")
    r = c.post("/api/analyze", data={"file": (open(FACE, "rb"), "face.jpg"), "age": "30"})
    d = r.get_json()
    check(r.status_code == 200 and len(d["indicators"]) == 6 and d["unavailable"], f"一般光源分析 6 項可見光指標 + UV 提示 ({r.status_code})")
    check(d["analysis_id"] is None and d["saved_to_history"] is False, "未登入未同意 → 不存")
    w = next(i for i in d["indicators"] if i["key"] == "wrinkles")
    from skinlens.analysis import wrinkle_seg
    expect = "ai_seg" if wrinkle_seg.available() else "model"
    check(w["source"] == expect, f"皺紋計分來源 = {expect} ({w['source']})")
    check(w["source"] != "ai_seg" or w["overlay"].startswith("data:image/jpeg"), "AI 皺紋線標示圖")
    check(not d["notices"] and not any(i["experimental"] for i in d["indicators"]), "一般光源沒有實驗性提示")
    r = c.post("/api/analyze", data={"file": (open(FACE, "rb"), "face.jpg"), "lighting": "uv"})
    d = r.get_json()
    check(r.status_code == 200 and sorted(i["key"] for i in d["indicators"]) == ["porphyrins", "uv_spots"],
          "UV 光源只分析紫外線色斑、紫質")
    check(all(i["experimental"] for i in d["indicators"]) and "實驗性" in d["notices"][0], "UV 結果標示實驗性 + 提醒")

    print("[會員]")
    check(c.post("/api/auth/register", json={"email": "bad", "password": "12345678"}).status_code == 400, "Email 格式錯 400")
    check(c.post("/api/auth/register", json={"email": "a@b.co", "password": "123"}).status_code == 400, "密碼太短 400")
    r = c.post("/api/auth/register", json={"email": "user@test.local", "password": "password1", "name": "小明"})
    check(r.status_code == 200 and r.get_json()["user"]["name"] == "小明", "註冊成功並登入")
    check(c.post("/api/auth/register", json={"email": "USER@test.local", "password": "password1"}).status_code == 409, "重複 Email 409 (不分大小寫)")
    check(c.put("/api/auth/profile", json={"age": 28, "gender": "F", "skin_type": "oily"}).status_code == 200, "更新個人資料")
    check(c.put("/api/auth/profile", json={"age": 300}).status_code == 400, "年齡超出範圍 400")
    for p in ["/history", "/account"]:
        check(c.get(p).status_code == 200, f"GET {p} 登入後 200")
    check(c.get("/admin").status_code == 302, "一般會員進後台 → 導回")
    check(c.get("/api/admin/stats").status_code == 403, "一般會員後台 API 403")

    print("[歷史紀錄]")
    ids = []
    for _ in range(2):
        r = c.post("/api/analyze", data={"file": (open(FACE, "rb"), "face.jpg")})
        d = r.get_json()
        check(r.status_code == 200 and d["saved_to_history"] and d["analysis_id"], "登入分析 → 存入紀錄")
        ids.append(d["analysis_id"])
    h = c.get("/api/history").get_json()
    check(len(h) == 2 and "wrinkles" in h[0]["scores"], f"紀錄列表 2 筆 ({len(h)})")
    det = c.get(f"/api/history/{ids[0]}").get_json()
    check(det["has_image"] and "overlay" not in det["indicators"][0], "紀錄詳情 (不含 base64 圖)")
    check(c.get(f"/api/history/{ids[0]}/image").status_code == 200, "紀錄照片可讀")
    check(c.get(f"/history/{ids[0]}").status_code == 200, "報告頁 200")
    check(c.get("/api/history/zzz").status_code == 404, "錯誤 id 404")

    other = app.test_client()
    other.post("/api/auth/register", json={"email": "other@test.local", "password": "password2"})
    check(other.get(f"/api/history/{ids[0]}").status_code == 404, "別人看不到我的紀錄")
    check(other.delete(f"/api/history/{ids[0]}").status_code == 404, "別人刪不掉我的紀錄")

    check(c.delete(f"/api/history/{ids[0]}").status_code == 200, "刪除單筆")
    check(len(c.get("/api/history").get_json()) == 1, "剩 1 筆")
    check(c.delete("/api/history").get_json()["deleted"] == 1, "刪除全部")
    c.post("/api/auth/logout")
    check(c.get("/api/auth/me").get_json()["user"] is None, "登出")

    print("[同意存入訓練資料]")
    r = c.post("/api/analyze", data={"file": (open(FACE, "rb"), "face.jpg"), "consent": "true"})
    check(r.get_json()["analysis_id"] is not None, "勾同意 → 存入")

    print("[後台]")
    check(c.post("/api/auth/login", json={"email": "admin@test.local", "password": "wrong"}).status_code == 401, "管理員密碼錯 401")
    check(c.post("/api/auth/login", json={"email": "admin@test.local", "password": "admin-pass-123"}).status_code == 200, "管理員登入")
    check(c.get("/admin").status_code == 200, "後台頁 200")
    s = c.get("/api/admin/stats").get_json()
    check(s["images"] == 1 and s["users"] == 3, f"統計 ({s['images']} 照片, {s['users']} 會員)")
    imgs = c.get("/api/admin/images").get_json()
    iid = imgs["items"][0]["id"]
    check(c.get(f"/api/admin/images/{iid}/file").status_code == 200, "照片檔可讀")
    check(c.post("/api/admin/annotations", json={"image_id": iid, "items": [{"indicator": "spots", "severity": 9}]}).status_code == 400, "嚴重度超出 400")
    r = c.post("/api/admin/annotations", json={"image_id": iid, "items": [{"indicator": "spots", "severity": 2}, {"indicator": "pores", "severity": 1}]})
    check(r.status_code == 200 and r.get_json()["saved"] == 2, "儲存標註")
    check(c.get("/api/admin/images?unlabeled=true").get_json()["total"] == 0, "未標註篩選")
    t = c.post("/api/admin/train").get_json()
    check(t["trained"] is False and "樣本不足" in t["metrics"]["spots"]["status"], "樣本不足不上線")
    check(len(c.get("/api/admin/models").get_json()) == 1, "訓練紀錄")
    csv = c.get("/api/admin/export.csv")
    check(csv.status_code == 200 and b"label_spots" in csv.data, "匯出 CSV")
    check(c.delete(f"/api/admin/images/{iid}").status_code == 200, "刪除照片")
    check(c.get("/api/admin/stats").get_json()["images"] == 0, "照片已刪")

    client.drop_database("skinlens_test")
    print(f"\n{'全部通過' if not check.failed else f'失敗 {check.failed} 項'}")
    return check.failed


def test_kaggle_import_helpers():
    """同一張臉的 Roboflow 副本要分到同一組；斑點依定義取暗斑 / 雀斑 / 痘疤最大值。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from import_kaggle_skin import original_of, labels_of
    assert original_of("oily_8a72f4-Copy_jpg.rf.63a8a6.jpg") == original_of("oily_8a72f4-Copy_jpg.rf.d434fc.jpg")
    assert original_of("normal_new_9bb5e1_jpg.rf.16c11a.jpg") == original_of("dry_9bb5e1_jpg.rf.abc123.jpg")
    lab = labels_of({"dark spots(0-5)": 1.0, "freckles(0-5)": 4.0, "post acne marks(0-5)": 2.0, "Open pores (0-5)": 3.0})
    assert lab["spots"] == 4.0 and lab["brown_spots"] == 1.0 and lab["pores"] == 3.0
    assert labels_of({"pigmentation(0-5)": 3.0})["brown_spots"] == 3.0


def test_all():
    assert run() == 0


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
