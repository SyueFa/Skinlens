"""皺紋分割：用 U-Net 讓模型自己從照片學會「畫出皺紋線」，再和人工遮罩逐像素對答案。

資料：scripts/prepare_wrinkle_seg.py 產生的 data/wrinkle_seg/ (臉部座標下的照片、人工遮罩、Frangi 對照組)
切分：train 800 / val 100 / test 100。只用 val 挑最好的訓練回合與門檻值，test 只在最後打一次分數。
模型：U-Net + ResNet34 編碼器 (ImageNet 預訓練)，Dice + BCE 損失 (皺紋只占約 1% 像素，類別極度不平衡)
硬體：CUDA (本機 RTX 5090)，bf16 混合精度

評分 (test 100 張)：
  Dice / IoU         模型畫的線和人工線逐像素重疊程度 (0 = 完全不重疊，1 = 完全一樣)
  容許 2px Dice      線條偏移 2 像素內也算對 (人工畫線本身就有幾像素誤差，細線很適合看這個)
  皺紋量相關 r        每張照片「模型畫的皺紋像素數」和「人工皺紋像素數」的相關 (= 拿來算分數時準不準)
對照組：現有傳統演算法 (Frangi 濾波) 在同樣 100 張上的同樣指標。

用法：uv run --extra train python scripts/train_wrinkle_seg.py [--epochs 60]
輸出：models/wrinkle_seg.pt (權重)、models/wrinkle_seg_metrics.json (分數)、data/wrinkle_seg/test_examples.jpg
"""
import argparse, json, random, sys, time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(d):
    face = cv2.imread(str(d / "face.png"))[..., ::-1].astype(np.float32) / 255.0
    mask = (cv2.imread(str(d / "mask.png"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
    return (face - MEAN) / STD, mask


class Crops(Dataset):
    """訓練用：每張照片隨機切 crop×crop，左右翻轉、輕微亮度對比變化 (不改色相，避免影響膚色)。"""
    def __init__(self, dirs, crop, repeat):
        self.items = [load(d) for d in dirs]
        self.crop, self.repeat = crop, repeat

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        x, y = self.items[i % len(self.items)]
        h, w = y.shape
        c = self.crop
        if h < c or w < c:
            ph, pw = max(0, c - h), max(0, c - w)
            x = np.pad(x, ((0, ph), (0, pw), (0, 0)), mode="reflect"); y = np.pad(y, ((0, ph), (0, pw)))
            h, w = y.shape
        r, s = random.randint(0, h - c), random.randint(0, w - c)
        x, y = x[r:r + c, s:s + c], y[r:r + c, s:s + c]
        if random.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = x * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(y))[None]


@torch.no_grad()
def predict(model, dirs, dev):
    """整張臉 (補到 32 的倍數) 推論 → 每張的皺紋機率圖。"""
    model.eval()
    out = []
    for d in dirs:
        x, _ = load(d)
        h, w = x.shape[:2]
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        xt = torch.from_numpy(np.pad(x, ((0, H - h), (0, W - w), (0, 0)), mode="reflect").transpose(2, 0, 1))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(xt).float())[0, 0, :h, :w]
        out.append(p.cpu().numpy())
    return out


def metrics(preds, gts, tol=2):
    """整體 (所有像素加總) 的 Dice / IoU、容許 tol 像素的 Dice、每張皺紋量相關。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    tp = fp = fn = 0
    tp_p = n_p = tp_g = n_g = 0
    amt_p, amt_g = [], []
    for p, g in zip(preds, gts):
        p, g = p.astype(bool), g.astype(bool)
        tp += (p & g).sum(); fp += (p & ~g).sum(); fn += (~p & g).sum()
        gd = cv2.dilate(g.astype(np.uint8), k) > 0
        pd = cv2.dilate(p.astype(np.uint8), k) > 0
        tp_p += (p & gd).sum(); n_p += p.sum()          # 容許偏移的 precision
        tp_g += (g & pd).sum(); n_g += g.sum()          # 容許偏移的 recall
        amt_p.append(p.sum()); amt_g.append(g.sum())
    prec, rec = tp_p / max(n_p, 1), tp_g / max(n_g, 1)
    return {"dice": round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4), "iou": round(float(tp / max(tp + fp + fn, 1)), 4),
            "tolerant_dice_2px": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 4),
            "tolerant_precision": round(float(prec), 4), "tolerant_recall": round(float(rec), 4),
            "amount_r": round(float(np.corrcoef(amt_p, amt_g)[0, 1]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "wrinkle_seg"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default="resnet34")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("找不到 CUDA GPU (預期使用本機 RTX 5090)")
    dev = "cuda"
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    data = Path(a.data)
    split = json.loads((data / "split.json").read_text(encoding="utf-8"))
    dirs = {k: [data / i for i in v] for k, v in split.items()}
    print(f"GPU: {torch.cuda.get_device_name(0)} | train {len(dirs['train'])} / val {len(dirs['val'])} / test {len(dirs['test'])}")

    loader = DataLoader(Crops(dirs["train"], a.crop, repeat=2), batch_size=a.batch, shuffle=True,
                        num_workers=4, persistent_workers=True, pin_memory=True, drop_last=True)
    val_gt = [load(d)[1] for d in dirs["val"]]
    model = smp.Unet(a.encoder, encoder_weights="imagenet", classes=1).to(dev)
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    pos_weight = torch.tensor([10.0], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader), pct_start=0.1)

    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(x)
            logit = logit.float()
            loss = dice_loss(logit, y) + F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        v = metrics([p > 0.5 for p in predict(model, dirs["val"], dev)], val_gt)
        if v["dice"] > best:
            best, best_state = v["dice"], {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {tot / len(loader):.4f}  val Dice {v['dice']:.4f}  容許2px {v['tolerant_dice_2px']:.4f}"
              f"  {'*' if v['dice'] >= best else ''}  ({time.time() - t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    val_prob = predict(model, dirs["val"], dev)
    thr = max(np.arange(0.2, 0.85, 0.05), key=lambda t: metrics([p > t for p in val_prob], val_gt)["dice"])
    test_gt = [load(d)[1] for d in dirs["test"]]
    test_prob = predict(model, dirs["test"], dev)
    unet = metrics([p > thr for p in test_prob], test_gt)
    frangi = metrics([cv2.imread(str(d / "frangi.png"), 0) > 127 for d in dirs["test"]], test_gt)
    print(f"\n門檻值 (由 val 決定)：{thr:.2f}")
    print(f"{'測試組 100 張':14s} {'Dice':>7s} {'IoU':>7s} {'容許2px Dice':>12s} {'皺紋量 r':>9s}")
    for name, m in (("U-Net (AI)", unet), ("Frangi (傳統)", frangi)):
        print(f"{name:14s} {m['dice']:7.4f} {m['iou']:7.4f} {m['tolerant_dice_2px']:12.4f} {m['amount_r']:9.4f}")

    torch.save({"state_dict": best_state, "encoder": a.encoder, "threshold": float(thr)}, ROOT / "models" / "wrinkle_seg.pt")
    (ROOT / "models" / "wrinkle_seg_metrics.json").write_text(json.dumps({
        "created_at": datetime.now().isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0),
        "encoder": a.encoder, "epochs": a.epochs, "best_val_dice": round(best, 4), "threshold": round(float(thr), 2),
        "split": {k: len(v) for k, v in split.items()}, "test": {"unet": unet, "frangi": frangi}}, indent=1), encoding="utf-8")

    # 測試組範例：照片 | 人工答案 | AI | 傳統 (綠 = 對、紅 = 多畫、藍 = 漏掉)
    rows = []
    for d, p, g in list(zip(dirs["test"], test_prob, test_gt))[:4]:
        face = cv2.imread(str(d / "face.png"))
        def paint(pred):
            v = (face * 0.55).astype(np.uint8); pred = pred.astype(bool); g_ = g.astype(bool)
            v[pred & g_] = (0, 220, 0); v[pred & ~g_] = (0, 0, 255); v[~pred & g_] = (255, 120, 0)
            return v
        gt_v = face.copy(); gt_v[g.astype(bool)] = (255, 255, 255)
        rows.append(np.hstack([face, gt_v, paint(p > thr), paint(cv2.imread(str(d / "frangi.png"), 0) > 127)]))
    h = min(r.shape[0] for r in rows)
    sheet = np.vstack([cv2.resize(r[:h], (r.shape[1] // 2, h // 2)) for r in rows])
    for i, t in enumerate(["photo", "human", "U-Net", "Frangi"]):
        cv2.putText(sheet, t, (i * 300 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(str(data / "test_examples.jpg"), sheet)
    print(f"→ models/wrinkle_seg.pt、models/wrinkle_seg_metrics.json、{data / 'test_examples.jpg'}")


if __name__ == "__main__":
    main()
