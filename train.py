"""Multi-part, per-part-severity classifier. Output: 6 parts x 3 classes (none/low/high), masked focal loss.
python train.py --csv labels.csv --arch convnext_small.fb_in22k_ft_in1k --size 384 --epochs 12 --out runs/cnx
Ablation flags: --loss {focal,ce}  --sampler {on,off}  --arch resnet50 / efficientnet_b1 ...
"""
import argparse, json, math, os, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, average_precision_score
from taxonomy import PARTS

P = len(PARTS)

class DS(Dataset):
    def __init__(s, df, train, size):
        s.df = df.reset_index(drop=True)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        s.tf = T.Compose(([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),  # no left/right classes -> flip is label-safe
                           T.ColorJitter(0.3, 0.3, 0.2)] if train else [T.Resize((size, size))]) + [T.ToTensor(), T.Normalize(mean, std)])
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]
        with Image.open(r["path"]) as im:
            img = im.convert("RGB")
        return s.tf(img), torch.tensor(r[PARTS].to_numpy(dtype=np.int64))

class Net(nn.Module):
    def __init__(s, arch, pretrained=True):
        super().__init__()
        s.b = timm.create_model(arch, pretrained=pretrained, num_classes=0, drop_rate=0.3)
        s.h = nn.Linear(s.b.num_features, P * 3)
    def forward(s, x): return s.h(s.b(x)).view(-1, P, 3)

def class_alpha(df, beta=0.99):
    """Effective-number weights per (part, class) from TRAIN labels, normalised to mean 1 per part."""
    a = np.ones((P, 3), np.float32)
    for j, p in enumerate(PARTS):
        n = np.array([(df[p] == c).sum() for c in range(3)], np.float64).clip(1)
        w = (1 - beta) / (1 - beta ** n); a[j] = (w / w.sum() * 3)
    return torch.tensor(a)

def loss_fn(logits, y, alpha, kind="focal", gamma=2.0):
    m = y >= 0
    if m.sum() == 0: return logits.sum() * 0
    lg, yy, pj = logits[m], y[m], m.nonzero()[:, 1]
    logp = F.log_softmax(lg, -1).gather(1, yy[:, None])[:, 0]
    w = alpha[pj, yy]
    return -(w * ((1 - logp.exp()) ** gamma if kind == "focal" else 1) * logp).mean()

def sampler_weights(df):
    freq = {(p, c): max(int((df[p] == c).sum()), 1) for p in PARTS for c in (1, 2)}
    w = []
    for _, r in df.iterrows():
        f = [freq[(p, int(r[p]))] for p in PARTS if r[p] in (1, 2)]
        w.append(1.0 / np.sqrt(min(f)) if f else None)
    valid = [x for x in w if x is not None]
    med = float(np.median(valid)) if valid else 1.0
    return [float(x) if x is not None else med for x in w]

@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); out, ys = [], []
    for x, y in loader:
        x = x.to(dev); pr = F.softmax(model(x).float(), -1) + F.softmax(model(x.flip(-1)).float(), -1)  # flip TTA
        out.append((pr / 2).cpu()); ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()

def tune_thresholds(pr, y):
    th = []
    cand = np.linspace(0.05, 0.95, 91)
    for j in range(P):
        m = y[:, j] >= 0
        if m.sum() == 0:
            th.append(0.5); continue
        t = (y[m, j] > 0).astype(int)
        if t.sum() == 0:
            th.append(0.5); continue
        s = 1 - pr[m, j, 0]
        th.append(float(cand[int(np.argmax([f1_score(t, s > c, zero_division=0) for c in cand]))]))
    return th

def _nanmean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if len(finite) else float("nan")

def _clean_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

def _load_state_dict(path, dev):
    try:
        return torch.load(path, map_location=dev, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=dev)

def evaluate(pr, y, th, sev_thr=0.5):
    res = {"parts": {}}; pred = np.zeros_like(y)
    for j, p in enumerate(PARTS):
        dam = (1 - pr[:, j, 0]) > th[j]
        high = pr[:, j, 2] / (pr[:, j, 1] + pr[:, j, 2] + 1e-9) > sev_thr
        pred[:, j] = np.where(dam, np.where(high, 2, 1), 0)
        m = y[:, j] >= 0; t = (y[m, j] > 0).astype(int)
        n_high = int((y[:, j] == 2).sum())
        d = {
            "n_damaged": int(t.sum()),
            "presence_F1": float(f1_score(t, dam[m], zero_division=0)) if m.sum() > 0 else float("nan"),
            "presence_AP": float(average_precision_score(t, 1 - pr[m, j, 0])) if (m.sum() > 0 and t.sum() > 0) else float("nan"),
        }
        both = m & (y[:, j] > 0)   # severity scored only where the part is truly damaged
        d["severity_F1_macro"] = float(f1_score(y[both, j], np.where(high[both], 2, 1), average="macro", zero_division=0)) if (both.sum() >= 2 and len(set(y[both, j].tolist())) > 1) else float("nan")
        d["high_recall"] = float(((pred[:, j] == 2) & (y[:, j] == 2)).sum() / n_high) if n_high > 0 else float("nan")
        res["parts"][p] = d
    res["presence_macro_F1"] = _nanmean([v["presence_F1"] for v in res["parts"].values()])
    res["presence_macro_AP"] = _nanmean([v["presence_AP"] for v in res["parts"].values()])
    res["severity_macro_F1"] = _nanmean([v["severity_F1_macro"] for v in res["parts"].values()])
    full = (y >= 0).all(1)
    res["exact_match_part+severity"] = float((pred[full] == y[full]).all(1).mean()) if full.any() else float("nan")
    res["part_set_exact_match"] = float(((pred[full] > 0) == (y[full] > 0)).all(1).mean()) if full.any() else float("nan")
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--arch", default="convnext_small.fb_in22k_ft_in1k")
    ap.add_argument("--size", type=int, default=384); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=24); ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--loss", choices=["focal", "ce"], default="focal")
    ap.add_argument("--sampler", choices=["on", "off"], default="on")
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default="runs/exp")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.csv)
    missing = [c for c in ["path", *PARTS] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "split" not in df:   # group split: put near-duplicate / same-car images in the same group via a 'group' column
        g = df["group"] if "group" in df else df["path"]
        if g.nunique() < 3:
            raise ValueError(f"Need at least 3 distinct groups for train/val/test split, got {g.nunique()}")
        tr_idx, rest = next(GroupShuffleSplit(1, train_size=0.70, random_state=a.seed).split(df, groups=g))
        va_idx, te_idx = next(GroupShuffleSplit(1, train_size=0.5, random_state=a.seed).split(df.iloc[rest], groups=g.iloc[rest]))
        df["split"] = "train"; df.loc[df.index[rest[va_idx]], "split"] = "val"; df.loc[df.index[rest[te_idx]], "split"] = "test"
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError(f"Empty split detected (train={len(tr)}, val={len(va)}, test={len(te)})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(a.arch, pretrained=not a.no_pretrained).to(dev); alpha = class_alpha(tr).to(dev)
    sm = WeightedRandomSampler(sampler_weights(tr), len(tr), replacement=True) if a.sampler == "on" else None
    mk = lambda d, t, **k: DataLoader(DS(d, t, a.size), batch_size=a.bs, num_workers=a.workers, **k)
    dl_tr = mk(tr, True, sampler=sm, shuffle=sm is None, drop_last=len(tr) > a.bs)
    dl_va, dl_te = mk(va, False), mk(te, False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=max(a.epochs * len(dl_tr), 2), pct_start=0.25)
    use_amp = dev == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp)
              if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        loss_sum, steps = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=use_amp):
                loss = loss_fn(model(x).float(), y, alpha, a.loss)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            loss_sum += float(loss.item()); steps += 1
        avg_loss = loss_sum / max(steps, 1)
        pr, y = predict(model, dl_va, dev); r = evaluate(pr, y, tune_thresholds(pr, y))
        score = float(np.nan_to_num(r["presence_macro_F1"], nan=0.0) + np.nan_to_num(r["severity_macro_F1"], nan=0.0))
        print(f"ep{ep} loss {avg_loss:.3f} val presenceF1 {r['presence_macro_F1']:.3f} sevF1 {r['severity_macro_F1']:.3f}")
        if score > best:
            best = score
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(_load_state_dict(out / "best.pt", dev))
    pr, y = predict(model, dl_va, dev); th = tune_thresholds(pr, y)      # thresholds from VAL only
    pt, yt = predict(model, dl_te, dev); res = evaluate(pt, yt, th)      # test touched once
    res.update(thresholds=dict(zip(PARTS, th)), args=vars(a))
    clean_res = _clean_json(res)
    with open(out / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(clean_res, f, indent=2, allow_nan=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"arch": a.arch, "size": a.size, "parts": PARTS, "thresholds": th, "sev_thr": 0.5}, f, indent=2)
    print(json.dumps({k: v for k, v in clean_res.items() if k != "parts"}, indent=2))

if __name__ == "__main__":
    main()
