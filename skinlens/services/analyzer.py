"""整條分析流程：照片 → 前處理 → 8 指標辨識 → (模型校正) → 同齡百分位 → 報告與建議 → 存入 MongoDB。"""
from __future__ import annotations
import base64
import hashlib
import uuid
import cv2
import numpy as np
from ..config import UPLOAD_DIR, PROC_DIR, HISTORY_DIR, MIN_PEERS
from ..db import subjects, images, analyses, now
from ..analysis.preprocess import load_image, preprocess, apply_color, device_model
from ..analysis import color
from ..analysis.indicators import (analyze_all, overlay, feature_vector, COLORS,
                                   detect_wrinkles, detect_spots, detect_brown_spots, detect_red_areas)
from ..analysis.knowledge import INDICATORS as KB, GENERAL_TIPS, DISCLAIMER, level_of
from . import trainer

IMAGE_KEYS = ("original", "preprocessed", "regions_preview")
EXPERIMENTAL = {"uv_spots", "porphyrins"}     # 需要 365nm UV 燈，尚未校準也沒有人工標註
# 側臉補充分析：魚尾紋 (皺紋) 與外側臉頰；臉要轉超過這個程度 (鼻尖離中線 |比例 − 0.5|) 才算側臉
SIDE_INDICATORS = {"wrinkles": detect_wrinkles, "spots": detect_spots,
                   "brown_spots": detect_brown_spots, "red_areas": detect_red_areas}
SIDE_MIN_TURN = 0.12
# 反光 (油光 + 過曝) 占皮膚 % 超過這個值就提醒洗臉重拍：FFHQ 參考族群的第 90 百分位約 12.8%
# (手機沒有交叉偏振片，表面油光只能靠使用者洗臉減少)
SHINE_WARN_PCT = 12.0
# 左右臉亮度差 (曝光級數，1 級 = 亮 2 倍) 超過這個值就提醒重拍：FFHQ 參考族群第 90 百分位約 1.18、
# 第 95 約 1.43；一般室內正面光的手機自拍約 0.4–0.9。大範圍明暗會做光照平整，但硬陰影 (鼻影、側光) 修不掉
LIGHT_WARN_STOPS = 1.3
# 正臉照片的頭部角度 (度) 超過就提醒重拍。網頁拍照流程會先引導到 ±10° 內才拍 (capture.js)，
# 這裡是手動上傳照片的保險。FFHQ 參考族群 (網路照片) 左右轉 P5–P95 約 −21~+17、上下 −13~+9；
# 一般手機自拍左右 1–3°、上下 −5~−11 (手機拿得比臉低)
POSE_WARN_YAW, POSE_WARN_PITCH = 15.0, 15.0
UV_NOTICE = ("紫外線色斑、紫質是實驗性功能：需要在 365nm UV 燈下拍攝，尚未經人工標註校正，結果僅供參考。"
             "如果照片是一般光線拍的，這兩項分數沒有意義。")


def _b64(img, q=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _session_color(pp, frames):
    """網頁拍照流程：用睜眼影格的眼白估光源，重新校色閉眼照片 (analysis/color.session_illuminant)。
    影格都不能用 (沒偵測到臉、眼白不可信) 時保留照片自己估的校色，並在 color 裡記下原因。"""
    fpps = []
    for fb in frames:
        try:
            fpps.append(preprocess(load_image(fb)))
        except ValueError:
            continue
    illum, info = color.session_illuminant(fpps, pp)
    info["frames_received"] = len(frames)
    if illum is not None:
        apply_color(pp, illum)
    pp["color"]["session"] = info


def pose_warning(pose):
    """臉沒有正對鏡頭時的提醒文字；正對或量不到 (Haar 備援偵測) 時回傳 None。"""
    if not pose:
        return None
    yaw, pitch = pose["yaw"], pose["pitch"]
    issues = []
    if abs(yaw) > POSE_WARN_YAW:
        issues.append(f"左右轉了約 {abs(yaw):.0f} 度")
    if pitch < -POSE_WARN_PITCH:
        issues.append(f"下巴抬高約 {-pitch:.0f} 度 (手機可能拿太低)")
    elif pitch > POSE_WARN_PITCH:
        issues.append(f"低頭約 {pitch:.0f} 度 (手機可能拿太高)")
    if not issues:
        return None
    return ("臉沒有正對鏡頭：" + "、".join(issues) + "，臉部分區會偏掉，一邊的臉頰或額頭被壓縮。"
            "建議手機與眼睛同高、臉正對鏡頭後重拍")


def extract(data: bytes, lighting="standard", frames=None, device=None):
    """影像 → (前處理結果, 各指標結果, 特徵向量)。模型校正也在這裡套用。
    frames：網頁拍照流程的睜眼影格 (JPEG bytes)，用來校色閉眼照片。
    device：照片沒有 EXIF 時 (網頁相機) 由前端提供的裝置名稱。"""
    pp = preprocess(load_image(data))
    pp["quality"]["device"] = device_model(data) or device
    if frames:
        _session_color(pp, frames)
    results = analyze_all(pp, lighting)
    feats = feature_vector(results, pp)
    shine = feats.get("specular.ratio", 0.0)
    pp["quality"]["shine_pct"] = round(shine, 1)
    if shine > SHINE_WARN_PCT:
        pp["quality"]["warnings"].append(
            f"臉上反光較多 (油光或過曝，占皮膚 {shine:.0f}%)，棕色斑、紅色區可能不準。"
            "建議洗臉擦乾、不擦保養品，約 10 分鐘後在柔和光線下重拍")
        pp["quality"]["ok"] = False
    light = pp["light"]
    pp["quality"]["light_lr_stops"] = round(light["lr_stops"], 2)
    pp["quality"]["light_range_stops"] = round(light["range_stops"], 2)
    if light["lr_stops"] > LIGHT_WARN_STOPS:
        pp["quality"]["warnings"].append(
            f"光線不均：左右臉亮度差約 {2 ** light['lr_stops']:.1f} 倍，陰影可能被誤判成斑點或皺紋。"
            "建議面對窗戶或燈光、讓光線從正前方照，避免側光與頭頂光後重拍")
        pp["quality"]["ok"] = False
    pose_msg = pose_warning(pp["quality"].get("pose"))
    if pose_msg:
        pp["quality"]["warnings"].append(pose_msg)
        pp["quality"]["ok"] = False
    learned = trainer.predict(feats)
    for k, r in results.items():
        if k in learned:
            r["severity_heuristic"] = r["severity"]
            r["severity"] = round(learned[k], 1)
            r["source"] = "model"
        elif r.get("method") == "unet":
            r["source"] = "ai_seg"          # AI 皺紋分割直接計分
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


def store_training_image(data, pp, feats, lighting, subject_code=None, source="upload", subject_info=None, group=None):
    """把照片存進訓練資料池 (images)。同一張照片 (sha256 相同) 只存一次。回傳 image _id。
    group：同一張原始照片的不同副本給相同的值，訓練時交叉驗證會把它們分在同一組 (避免資料洩漏)。"""
    sha = hashlib.sha256(data).hexdigest()
    found = images.find_one({"sha256": sha}, {"_id": 1})
    if found:
        images.update_one({"_id": found["_id"]}, {"$set": {"features": feats, **({"group": group} if group else {})}})
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
        "group": group,
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
            "count_split": ({"棕色": r["brown_count"], "紅色": r["red_count"]} if "brown_count" in r else None),
            "source": r["source"], "percentile": pct.get(k), "experimental": k in EXPERIMENTAL,
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
        "lighting": lighting, "quality": pp["quality"], "color": pp["color"], "overall_score": overall, "summary": summary,
        "original": _b64(pp["face_raw"]), "preprocessed": _b64(pp["face"]),
        "regions_preview": _b64(_region_preview(pp)),
        "indicators": indicators, "general_tips": GENERAL_TIPS, "disclaimer": DISCLAIMER,
        "unavailable": ([] if lighting == "uv" else ["紫外線色斑、紫質需要 UV 光源照片，請另外上傳並選擇「UV 光源」"]),
        "notices": ([UV_NOTICE] if lighting == "uv" else []),
    }


def _yaw_ratio(pp):
    """鼻尖在左右臉緣之間的位置：0.5 = 正臉；越偏離 0.5 轉得越多。"""
    kp = pp["kp"]
    l, r = float(kp["face_l"][0]), float(kp["face_r"][0])
    return (float(kp["nose"][0]) - l) / max(r - l, 1e-6)


def analyze_sides(sides, illuminant=None):
    """側臉補充分析 (網頁拍照流程的左、右轉照片)，像 VISIA 的側面照：
    看魚尾紋 (皺紋) 與外側臉頰的斑點、棕色斑、紅色區。主分數仍來自正臉照片 —— 嚴重度曲線是用
    正臉照片校準的，側臉數值只列出量測值與標示位置，不換算成分數。
    illuminant：沿用正臉照片的光源 (同一台相機、只隔幾秒)。"""
    out = []
    for b in sides:
        try:
            sp = preprocess(load_image(b), illuminant=illuminant)
        except ValueError:
            out.append({"error": "側臉照片偵測不到臉部"})
            continue
        turn = _yaw_ratio(sp) - 0.5
        # 鼻尖偏向畫面某一邊時，看得到的是另一邊的臉頰。網頁相機存的影像不鏡像，
        # 畫面右半 = 本人左臉。分區名稱 *_l / *_r 是畫面的左 / 右
        suffix, side = ("r", "左臉") if turn < 0 else ("l", "右臉")
        item = {"side": side, "turn": round(abs(turn) * 2, 2), "original": _b64(sp["face"]),
                "metrics": {}, "overlays": {}}
        if abs(turn) < SIDE_MIN_TURN:
            item["warning"] = "臉轉得不夠，側面看不清楚；下次請再多轉一點 (約 45 度)"
        for k, fn in SIDE_INDICATORS.items():
            r = fn(sp)
            item["metrics"][k] = {"name": KB[k]["name"], "count": r["count"],
                                  "crow": r["by_region"].get(f"crow_{suffix}"),
                                  "cheek": r["by_region"].get(f"cheek_{suffix}")}
            item["overlays"][k] = _b64(overlay(sp["face"], r["mask"], COLORS[k]))
        out.append(item)
    return out


def strip_images(report):
    """存進歷史紀錄的報告不含 base64 圖片 (太大)，只留分數與建議。"""
    r = {k: v for k, v in report.items() if k not in IMAGE_KEYS}
    r["indicators"] = [{k: v for k, v in i.items() if k != "overlay"} for i in report["indicators"]]
    if "sides" in report:
        r["sides"] = [{k: v for k, v in s.items() if k not in ("original", "overlays")} for s in report["sides"]]
    return r


def analyze(data: bytes, lighting="standard", age=None, gender=None, skin_type=None, consent=False, user=None,
            frames=None, device=None, sides=None):
    """網站上傳分析。會員 → 存入歷史紀錄；勾選同意 → 照片另存訓練資料池。
    frames：網頁拍照流程的睜眼影格，用來校色閉眼照片。sides：左、右轉的側臉照片 (補充分析)。"""
    pp, results, feats = extract(data, lighting, frames, device)
    report = build_report(pp, results, age, lighting)
    if sides:
        ill = pp["color"].get("illuminant_bgr")
        report["sides"] = analyze_sides(sides, ill)

    image_id = None
    if consent:
        image_id = store_training_image(data, pp, feats, lighting,
                                        subject_info={"age": age, "gender": gender, "skin_type": skin_type})
    analysis_id = None
    if user or consent:
        doc = {"user_id": user["_id"] if user else None, "image_id": image_id, "lighting": lighting,
               "age": age, "gender": gender, "skin_type": skin_type, "device": pp["quality"]["device"],
               "overall_score": report["overall_score"], "summary": report["summary"],
               "results": {k: {"severity": r["severity"], "raw": r["raw"], "count": r["count"], "source": r["source"]}
                           for k, r in results.items()},
               "model_version": "model" if any(r["source"] in ("model", "ai_seg") for r in results.values()) else "heuristic",
               "report": strip_images(report), "created_at": now()}
        analysis_id = analyses.insert_one(doc).inserted_id
        if user:
            thumb = HISTORY_DIR / f"{analysis_id}.jpg"
            cv2.imwrite(str(thumb), pp["face"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            analyses.update_one({"_id": analysis_id}, {"$set": {"thumb_path": str(thumb)}})
    report["analysis_id"] = str(analysis_id) if analysis_id else None
    report["saved_to_history"] = bool(user)
    return report
