"""從同一份定義產生 draw.io 檔 + 預覽頁用的 SVG，確保兩邊一致。
用法：python gen_diagrams.py <drawio 輸出路徑> <svg 片段輸出資料夾>"""
import sys, html
from pathlib import Path
from xml.sax.saxutils import escape

# layer → (drawio fill, drawio stroke)
PAL = {
    "client": ("#e8f0fb", "#3b6fb6"), "server": ("#e6f4ec", "#2f8a57"),
    "engine": ("#fdf1e3", "#c07a1e"), "data": ("#e3f3f3", "#1f8a8a"),
    "ext": ("#f2f2f5", "#6b6f80"), "neutral": ("#ffffff", "#5b6770"),
    "yes": ("#e6f4ec", "#2f8a57"), "no": ("#fbe9e7", "#b5452f"),
}


class Diagram:
    def __init__(self, name, w, h):
        self.name, self.w, self.h = name, w, h
        self.nodes, self.edges = [], []

    def band(self, id, x, y, w, h, title, sub, layer):
        self.nodes.append(dict(id=id, x=x, y=y, w=w, h=h, title=title, sub=sub, layer=layer, kind="band"))

    def box(self, id, x, y, w, h, title, sub="", layer="neutral", kind="box"):
        self.nodes.append(dict(id=id, x=x, y=y, w=w, h=h, title=title, sub=sub, layer=layer, kind=kind))

    def edge(self, a, b, label="", dashed=False, pts=None, sa=None, ta=None):
        self.edges.append(dict(a=a, b=b, label=label, dashed=dashed, pts=pts, sa=sa, ta=ta))

    def n(self, id):
        return next(n for n in self.nodes if n["id"] == id)

    # ---------- draw.io ----------
    def drawio(self, page_id):
        cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        for n in self.nodes:
            fill, stroke = PAL[n["layer"]]
            dash = "dashed=1;" if n["layer"] == "ext" else ""
            if n["kind"] == "band":
                val = f'<b style="font-size:15px">{escape(n["title"])}</b><br><span style="font-size:11px;color:#5b6770">{escape(n["sub"])}</span>'
                style = (f"rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};{dash}"
                         "verticalAlign=top;align=left;spacingLeft=12;spacingTop=6;fontFamily=Noto Sans TC;")
            elif n["kind"] == "diamond":
                val = f'<b>{escape(n["title"])}</b>'
                style = f"rhombus;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontFamily=Noto Sans TC;"
            elif n["kind"] == "db":
                val = f'<b>{escape(n["title"])}</b>' + (f'<br><span style="font-size:11px">{escape(n["sub"])}</span>' if n["sub"] else "")
                style = f"shape=cylinder3;boundedLbl=1;size=10;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={stroke};fontFamily=Noto Sans TC;"
            elif n["kind"] == "terminal":
                val = f'<b>{escape(n["title"])}</b>'
                style = f"rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};fontFamily=Noto Sans TC;"
            else:
                val = f'<b>{escape(n["title"])}</b>' + (f'<br><span style="font-size:11px;color:#5b6770">{escape(n["sub"])}</span>' if n["sub"] else "")
                style = f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor={stroke};{dash}fontFamily=Noto Sans TC;"
            cells.append(f'<mxCell id="{n["id"]}" value="{html.escape(val)}" style="{style}" vertex="1" parent="1">'
                         f'<mxGeometry x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{n["h"]}" as="geometry"/></mxCell>')
        for i, e in enumerate(self.edges):
            st = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeColor=#5b6770;fontFamily=Noto Sans TC;fontSize=11;labelBackgroundColor=#ffffff;"
            if e["dashed"]: st += "dashed=1;"
            for key, anchor in (("exit", e["sa"]), ("entry", e["ta"])):
                if anchor:
                    st += f"{key}X={anchor[0]};{key}Y={anchor[1]};{key}Dx=0;{key}Dy=0;"
            pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in (e["pts"] or []))
            arr = ('<Array as="points">' + pts + "</Array>") if pts else ""
            geo = f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry>'
            cells.append(f'<mxCell id="{page_id}e{i}" value="{html.escape(e["label"])}" style="{st}" edge="1" parent="1" '
                         f'source="{e["a"]}" target="{e["b"]}">{geo}</mxCell>')
        return (f'<diagram id="{page_id}" name="{escape(self.name)}"><mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" '
                f'guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.w}" '
                f'pageHeight="{self.h}" math="0" shadow="0"><root>{"".join(cells)}</root></mxGraphModel></diagram>')

    # ---------- SVG ----------
    def _anchor(self, n, a):
        return (n["x"] + n["w"] * a[0], n["y"] + n["h"] * a[1])

    def svg(self):
        o = [f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{escape(self.name)}" xmlns="http://www.w3.org/2000/svg">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" class="arrowhead"/></marker></defs>']
        for n in self.nodes:
            L, x, y, w, h = n["layer"], n["x"], n["y"], n["w"], n["h"]
            cx = x + w / 2
            if n["kind"] == "band":
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" class="band {L}"/>')
                o.append(f'<text x="{x+14}" y="{y+24}" class="band-title {L}">{escape(n["title"])}</text>')
                o.append(f'<text x="{x+14}" y="{y+42}" class="band-sub">{escape(n["sub"])}</text>')
                continue
            if n["kind"] == "diamond":
                o.append(f'<polygon points="{cx},{y} {x+w},{y+h/2} {cx},{y+h} {x},{y+h/2}" class="node {L}"/>')
            elif n["kind"] == "db":
                ry = 9
                o.append(f'<path d="M{x},{y+ry} a{w/2},{ry} 0 0 0 {w},0 v{h-2*ry} a{w/2},{ry} 0 0 1 {-w},0 z" class="node {L}"/>')
                o.append(f'<ellipse cx="{cx}" cy="{y+ry}" rx="{w/2}" ry="{ry}" class="node {L}"/>')
            else:
                rx = h / 2 if n["kind"] == "terminal" else 8
                cls = f"node {L}" + (" filled" if n["kind"] == "terminal" else "")
                o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{cls}"/>')
            ty = y + h / 2 + (4 if n["kind"] == "db" else 0)
            if n["sub"]:
                o.append(f'<text x="{cx}" y="{ty-3}" class="node-title" text-anchor="middle">{escape(n["title"])}</text>')
                o.append(f'<text x="{cx}" y="{ty+13}" class="node-sub" text-anchor="middle">{escape(n["sub"])}</text>')
            else:
                o.append(f'<text x="{cx}" y="{ty+5}" class="node-title" text-anchor="middle">{escape(n["title"])}</text>')
        for e in self.edges:
            a, b = self.n(e["a"]), self.n(e["b"])
            p0 = self._anchor(a, e["sa"] or (0.5, 1)); p1 = self._anchor(b, e["ta"] or (0.5, 0))
            mids = list(e["pts"] or [])
            if not mids and p0[0] != p1[0] and p0[1] != p1[1]:
                my_ = (p0[1] + p1[1]) / 2
                mids = [(p0[0], my_), (p1[0], my_)]
            pts = [p0] + mids + [p1]
            d = "M" + " L".join(f"{px},{py}" for px, py in pts)
            o.append(f'<path d="{d}" class="edge{" dashed" if e["dashed"] else ""}" marker-end="url(#arr)"/>')
            if e["label"]:
                segs = list(zip(pts, pts[1:]))
                (ax, ay), (bx, by) = max(segs, key=lambda sg: abs(sg[0][0]-sg[1][0]) + abs(sg[0][1]-sg[1][1]))
                mx, my = (ax + bx) / 2, (ay + by) / 2
                tw = len(e["label"]) * 11 + 12
                o.append(f'<rect x="{mx-tw/2}" y="{my-10}" width="{tw}" height="20" rx="4" class="edge-label-bg"/>')
                o.append(f'<text x="{mx}" y="{my+4}" class="edge-label" text-anchor="middle">{escape(e["label"])}</text>')
        o.append("</svg>")
        return "\n".join(o)


# ======================================================== 圖 1：系統架構
A = Diagram("系統架構圖", 1220, 800)
A.band("bA", 20, 20, 900, 160, "使用者端（瀏覽器）", "HTML · CSS · JavaScript · jQuery · Bootstrap 5", "client")
for i, (t, s) in enumerate([("首頁 / 拍照上傳", "拍攝規範提示"), ("分析報告頁", "8 指標 · 問題區域 · 建議"),
                            ("歷史紀錄", "前後比較趨勢圖"), ("指標百科", "代表什麼 · 怎麼改善"), ("後台", "標註 · 訓練 · 統計")]):
    A.box(f"p{i}", 40 + i * 176, 82, 160, 72, t, s, "client")

A.band("bB", 20, 240, 900, 150, "Flask 後端（Python 3.12）", "Blueprint 路由 · JSON API · 管理員登入驗證", "server")
for i, (t, s) in enumerate([("分析 API", "POST /api/analyze"), ("歷史 API", "GET /api/history"),
                            ("知識庫 API", "GET /api/knowledge"), ("後台 API", "/api/admin/*（需登入）")]):
    A.box(f"s{i}", 40 + i * 220, 300, 200, 64, t, s, "server")

A.band("bC", 20, 450, 900, 140, "影像分析引擎", "OpenCV · MediaPipe · scikit-image · scikit-learn", "engine")
eng = [("前處理", "MediaPipe 478 點 · 白平衡 · 分區"), ("8 指標辨識", "斑點 · 皺紋 · 紋理 · 毛孔 …"),
       ("模型校正", "Gradient Boosting"), ("報告產生", "嚴重度 · 百分位 · 建議")]
for i, (t, s) in enumerate(eng):
    A.box(f"c{i}", 40 + i * 220, 508, 200, 60, t, s, "engine")
for i in range(3):
    A.edge(f"c{i}", f"c{i+1}", sa=(1, 0.5), ta=(0, 0.5))

A.band("bD", 20, 650, 900, 130, "資料層", "MongoDB（pymongo）· 本機 / MongoDB Atlas", "data")
for i, (t, s) in enumerate([("subjects", "受測者"), ("images", "照片 · 特徵"), ("annotations", "人工標註"),
                            ("analyses", "分析紀錄"), ("model_versions", "模型版本")]):
    A.box(f"d{i}", 40 + i * 150, 704, 136, 58, t, s, "data", kind="db")
A.box("files", 790, 704, 112, 58, "照片檔案", "uploads / processed", "data")

A.edge("bA", "bB", "AJAX · JSON", sa=(0.5, 1), ta=(0.5, 0))
A.edge("bB", "bC", "呼叫分析", sa=(0.5, 1), ta=(0.5, 0))
A.edge("bC", "bD", "pymongo 讀寫", sa=(0.5, 1), ta=(0.5, 0))

A.band("bE", 950, 20, 250, 760, "外部資源", "資料集 · 模型 · 部署", "ext")
A.box("x1", 970, 82, 210, 64, "FFHQ-Wrinkle", "1,000 張人工皺紋標註", "ext")
A.box("x2", 970, 190, 210, 64, "匯入腳本", "遮罩 → 皺紋分數", "ext")
A.box("x4", 970, 508, 210, 60, "訓練模型", "交叉驗證 · 優於基準才上線", "ext")
A.box("x5", 970, 680, 210, 80, "部署（選用）", "Render + MongoDB Atlas", "ext")
A.edge("x1", "x2")
A.edge("x2", "d2", "寫入標註", dashed=True, sa=(0, 0.5), ta=(0.5, 1), pts=[(940, 222), (940, 790), (378, 790)])
A.edge("x4", "c2", "更新模型", dashed=True, sa=(0, 0.5), ta=(0.5, 1), pts=[(930, 538), (930, 582), (580, 582)])

# ======================================================== 圖 2：分析與訓練流程
F = Diagram("分析與訓練流程", 1000, 1000)
F.band("fa", 20, 20, 520, 960, "分析流程（每次上傳）", "使用者在網頁上傳照片後，後端依序執行", "client")
steps = [("t0", "上傳照片", "", "terminal"), ("f1", "讀取影像", "EXIF 轉正 · 縮圖", "box"),
         ("f2", "臉部偵測與對齊", "MediaPipe 478 點 → 依雙眼轉正 · 寬 600px", "box"),
         ("f3", "色彩校正", "Shades-of-Gray 白平衡 · CLAHE", "box"),
         ("f4", "膚色遮罩 + 切 10 分區", "排除頭髮眼眉嘴", "box"),
         ("f5", "品質檢查", "模糊 · 太暗 · 過曝 · 臉太小 → 警告", "box"),
         ("f6", "8 項指標辨識", "原始量測值 · 0~100 嚴重度 · 病灶遮罩", "box"),
         ("q1", "有已上線模型？", "", "diamond")]
y = 76
for id, t, s, k in steps:
    h = 44 if k == "terminal" else (80 if k == "diamond" else 58)
    w = 200 if k == "terminal" else (220 if k == "diamond" else 300)
    F.box(id, 180 - (w - 300) / 2 if k != "box" else 180, y, w if k != "box" else 300, h, t, s,
          "neutral" if k != "terminal" else "client", kind=k)
    y += h + 26
for a, b in zip([s[0] for s in steps], [s[0] for s in steps][1:]):
    F.edge(a, b)
F.box("m1", 50, y + 10, 210, 56, "模型分數", "用人工標註校正過", "yes")
F.box("m2", 300, y + 10, 210, 56, "演算法經驗分數", "尚未校正的指標", "no")
F.edge("q1", "m1", "是", sa=(0, 0.5), ta=(0.5, 0), pts=[(155, F.n("q1")["y"] + 40)])
F.edge("q1", "m2", "否", sa=(1, 0.5), ta=(0.5, 0), pts=[(405, F.n("q1")["y"] + 40)])
y += 96
F.box("f7", 180, y, 300, 58, "同齡百分位", "同年齡 ±5 歲 · 至少 10 筆", "neutral")
F.edge("m1", "f7", ta=(0.25, 0)); F.edge("m2", "f7", ta=(0.75, 0))
F.box("f8", 180, y + 84, 300, 58, "報告 + 改善建議", "依嚴重度給保養建議 · 何時就醫", "neutral")
F.box("t1", 230, y + 168, 200, 44, "顯示報告 · 存入 MongoDB", "", "client", kind="terminal")
F.edge("f7", "f8"); F.edge("f8", "t1")

F.band("fb", 570, 20, 410, 960, "訓練流程（後台）", "標註累積後，管理員一鍵訓練", "server")
tsteps = [("u0", "資料來源", "公開資料集 · 後台人工標註", "box"),
          ("u1", "匯入 MongoDB", "images + annotations", "box"),
          ("u2", "整理訓練資料", "特徵向量 X · 人工分數 y（多人取平均）", "box"),
          ("u3", "Gradient Boosting 迴歸", "每個指標各一個模型", "box"),
          ("u4", "5 折交叉驗證", "MAE · Pearson r", "box"),
          ("q2", "誤差比猜平均低 10%？", "", "diamond")]
y = 76
for id, t, s, k in tsteps:
    h = 80 if k == "diamond" else 58
    w = 240 if k == "diamond" else 330
    F.box(id, 610 + (330 - w) / 2, y, w, h, t, s, "neutral", kind=k)
    y += h + 30
for a, b in zip([s[0] for s in tsteps], [s[0] for s in tsteps][1:]):
    F.edge(a, b)
F.box("v1", 590, y + 10, 175, 58, "上線", "分析改用模型分數", "yes")
F.box("v2", 785, y + 10, 175, 58, "不上線", "沿用演算法分數", "no")
F.edge("q2", "v1", "是", sa=(0, 0.5), ta=(0.5, 0), pts=[(677, F.n("q2")["y"] + 40)])
F.edge("q2", "v2", "否", sa=(1, 0.5), ta=(0.5, 0), pts=[(872, F.n("q2")["y"] + 40)])
F.box("v3", 610, y + 100, 330, 58, "記錄模型版本", "model_versions：指標 · 樣本數 · 誤差", "neutral")
F.edge("v1", "v3", ta=(0.25, 0)); F.edge("v2", "v3", ta=(0.75, 0))

drawio_out, svg_dir = Path(sys.argv[1]), Path(sys.argv[2])
drawio_out.parent.mkdir(parents=True, exist_ok=True)
drawio_out.write_text('<mxfile host="app.diagrams.net">' + A.drawio("arch") + F.drawio("flow") + "</mxfile>", encoding="utf-8")
(svg_dir / "arch.svg").write_text(A.svg(), encoding="utf-8")
(svg_dir / "flow.svg").write_text(F.svg(), encoding="utf-8")
print("ok", drawio_out)
