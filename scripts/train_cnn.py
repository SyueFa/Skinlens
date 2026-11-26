"""
第二階段：直接從照片訓練 CNN (多任務：一個骨幹網路同時輸出 8 個指標分數)。
資料量建議 每個指標 > 300 張標註後再用；少量資料先用網頁上的「訓練模型」(Gradient Boosting) 即可。

安裝：uv add torch torchvision timm
執行：uv run python scripts/train_cnn.py --epochs 30
輸出：models/skin_cnn.pt

流程：
  讀資料庫 (前處理後的臉部照片 + 標註平均分數) → 資料擴增 → EfficientNet-B0 (ImageNet 預訓練) 微調
  → 遮罩式 loss (沒標註的指標不計 loss) → 用 val split 選最佳 epoch
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
import torchvision.transforms as T
import timm
from skinlens.db import images, annotations, INDICATORS
from skinlens.config import MODEL_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--size", type=int, default=384)   # 細紋、毛孔需要高解析度，不要縮太小
ap.add_argument("--lr", type=float, default=3e-4)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_rows(split):
    """從 MongoDB 取出某個切分 (train/val/test) 的照片與各指標平均分數 (0~1)。"""
    rows = []
    for im in images.find({"split": split}, {"processed_path": 1}):
        y = np.full(len(INDICATORS), np.nan, np.float32)
        for i, ind in enumerate(INDICATORS):
            s = [a["score"] for a in annotations.find({"image_id": im["_id"], "indicator": ind}, {"score": 1})
                 if a.get("score") is not None]
            if s:
                y[i] = np.mean(s) / 100
        if not np.isnan(y).all():
            rows.append((im["processed_path"], y))
    return rows


class SkinDS(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        # 注意：不要做色彩抖動 (會改變紅色區/棕色斑的答案)，只做幾何與輕微亮度
        aug = [T.RandomResizedCrop(args.size, scale=(0.85, 1.0), ratio=(0.8, 1.0)), T.RandomRotation(5),
               T.ColorJitter(brightness=0.1)] if train else [T.Resize((args.size, args.size))]
        self.tf = T.Compose(aug + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        p, y = self.rows[i]
        return self.tf(PILImage.open(p).convert("RGB")), torch.tensor(y)


def masked_loss(pred, y):
    m = ~torch.isnan(y)
    return nn.functional.smooth_l1_loss(pred[m], y[m]) if m.any() else pred.sum() * 0


train_rows, val_rows = load_rows("train"), load_rows("val")
print(f"train {len(train_rows)}  val {len(val_rows)}  device {dev}")
if len(train_rows) < 50:
    sys.exit("訓練資料太少 (<50)，先多收集/標註，或用網頁的 Gradient Boosting 模型")

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(INDICATORS)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
tl = DataLoader(SkinDS(train_rows, True), batch_size=args.bs, shuffle=True, num_workers=2)
vl = DataLoader(SkinDS(val_rows, False), batch_size=args.bs)
best = 1e9
for ep in range(args.epochs):
    model.train()
    for x, y in tl:
        x, y = x.to(dev), y.to(dev)
        loss = masked_loss(torch.sigmoid(model(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    model.eval(); errs = [[] for _ in INDICATORS]
    with torch.no_grad():
        for x, y in vl:
            p = torch.sigmoid(model(x.to(dev))).cpu()
            for i in range(len(INDICATORS)):
                m = ~torch.isnan(y[:, i])
                errs[i] += (100 * (p[m, i] - y[m, i]).abs()).tolist()
    maes = {ind: round(float(np.mean(e)), 2) for ind, e in zip(INDICATORS, errs) if e}
    mean_mae = np.mean(list(maes.values())) if maes else 1e9
    print(f"epoch {ep+1:02d}  val MAE {mean_mae:.2f}  {maes}")
    if mean_mae < best:
        best = mean_mae
        torch.save({"state": model.state_dict(), "indicators": INDICATORS, "size": args.size, "val_mae": maes},
                   MODEL_DIR / "skin_cnn.pt")
print(f"best val MAE {best:.2f} → models/skin_cnn.pt")
