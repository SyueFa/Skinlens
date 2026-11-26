"""SkinLens 入口。

本地開發：uv run python app.py          → http://localhost:5000
雲端部署 (Render)：同一個指令，PORT 由平台提供
"""
import os
from skinlens import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1" and "PORT" not in os.environ
    app.run(host="0.0.0.0" if "PORT" in os.environ else "127.0.0.1", port=port, debug=debug)
