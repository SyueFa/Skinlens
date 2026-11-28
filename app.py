"""SkinLens 入口。

本地開發：uv run python app.py          → http://localhost:5000
手機測試 (區網 HTTPS)：HTTPS=1 uv run python app.py → https://<這台電腦的區網 IP>:5000
    手機瀏覽器只在 HTTPS 下才允許開鏡頭。憑證放在 data/certs/ (cert.pem、key.pem，自簽，不進 git)，
    手機第一次開會出現「不安全」警告，按「繼續前往」即可。這個模式關掉除錯模式 (除錯主控台不能開放給區網)。
    勾選「同意」的照片存到 informations/capture_<時間>/ (CONSENT_SAVE_DIR 可改)，不進資料庫。
雲端部署 (Render)：同一個指令，PORT 由平台提供
"""
import os
from pathlib import Path

if os.environ.get("HTTPS") == "1":
    # 手機測試：勾選「同意」的照片存到 informations/ (不進 git、不進資料庫)，方便跨次比較
    os.environ.setdefault("CONSENT_SAVE_DIR", "informations")

from skinlens import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if os.environ.get("HTTPS") == "1":
        certs = Path(__file__).resolve().parent / "data" / "certs"
        app.run(host="0.0.0.0", port=port, debug=False,
                ssl_context=(str(certs / "cert.pem"), str(certs / "key.pem")))
    else:
        debug = os.environ.get("FLASK_DEBUG", "1") == "1" and "PORT" not in os.environ
        app.run(host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=port, debug=debug)
