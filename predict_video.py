"""Run the trained model on a video: sample frames, aggregate to per-part damage + severity.
python predict_video.py --video car.mp4 --run runs/cnx --every-sec 0.5 --out result.json
Video has no labels here -> use for qualitative demo / stress test unless you hand-annotate it."""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from train import Net, _load_state_dict
from taxonomy import PARTS

def predict_video(video_path, run_dir, every_sec=0.5):
    if every_sec <= 0:
        raise ValueError(f"--every-sec must be > 0, got {every_sec}")
    run = Path(run_dir)
    with open(run / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(meta["arch"], pretrained=False).to(dev)
    model.load_state_dict(_load_state_dict(run / "best.pt", dev))
    model.eval()
    tf = T.Compose([T.Resize((meta["size"],) * 2), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    probs, i = [], 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps and fps > 0 and np.isfinite(fps)) else 30.0
        step = max(int(round(fps * every_sec)), 1)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                x = tf(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))[None].to(dev)
                with torch.no_grad():
                    pr_f = (F.softmax(model(x).float(), -1) + F.softmax(model(x.flip(-1)).float(), -1)) / 2
                    probs.append(pr_f[0].cpu().numpy())
            i += 1
    finally:
        cap.release()
    if not probs:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    pr = np.stack(probs)                                   # frames x 6 x 3
    sev_thr = float(meta.get("sev_thr", 0.5))
    res = {"n_frames": int(len(pr))}
    for j, p in enumerate(PARTS):
        dmg = 1 - pr[:, j, 0]; k = max(1, len(dmg) // 5)
        score = float(np.sort(dmg)[-k:].mean())            # mean of top-20% frames (robust to a few bad views)
        top = np.argsort(dmg)[-k:]
        hi = float((pr[top, j, 2] / (pr[top, j, 1] + pr[top, j, 2] + 1e-9)).mean())
        damaged = bool(score > meta["thresholds"][j])
        severity = ("high" if hi > sev_thr else "low") if damaged else "none"
        res[p] = {"damage_score": score, "damaged": damaged, "severity": severity, "p_high": hi}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--every-sec", type=float, default=0.5); ap.add_argument("--out", default="video_result.json")
    a = ap.parse_args()
    res = predict_video(a.video, a.run, a.every_sec)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
