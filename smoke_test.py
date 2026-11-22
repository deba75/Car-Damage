"""End-to-end synthetic smoke test for taxonomy -> build_labels -> train -> predict_video."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from taxonomy import PARTS, map_part, reset_unmapped, unmapped_names


def test_taxonomy():
    reset_unmapped()
    cases = {
        "front_bumper": "front",
        "rear_bumper": "rear",
        "front_left_door": "door",
        "rear_window": "window",
        "front_left_fender": "corner",
        "rear_quarter_panel": "corner",
        "front_wheel": "other",
        "rear_license_plate": "other",
        "headlight": "front",
        "tailgate": "rear",
        "mystery_part": "other",
    }
    for raw, expected in cases.items():
        got = map_part(raw)
        assert got == expected, f"map_part({raw!r}) = {got!r}, expected {expected!r}"
    assert unmapped_names() == ["mystery_part"], f"unexpected unmapped: {unmapped_names()}"
    print("[ok] taxonomy mapping")


def make_synthetic_yolo_dataset(root: Path, n_images: int = 12):
    img_dir = root / "images" / "train"
    lp_dir = root / "labels_parts" / "train"
    ld_dir = root / "labels_damage" / "train"
    for d in (img_dir, lp_dir, ld_dir):
        d.mkdir(parents=True, exist_ok=True)

    (root / "parts.yaml").write_text(
        "names:\n  0: front_bumper\n  1: rear_bumper\n  2: front_door\n  3: front_fender\n  4: windshield\n  5: front_wheel\n",
        encoding="utf-8",
    )
    (root / "damage.yaml").write_text(
        "names:\n  0: scratch\n  1: dent\n  2: crack\n",
        encoding="utf-8",
    )

    rng = np.random.default_rng(42)
    for i in range(n_images):
        img = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"car_{i:03d}.png"), img)
        # Put two parts per image
        p1 = i % 6
        p2 = (i + 2) % 6
        (lp_dir / f"car_{i:03d}.txt").write_text(
            f"{p1} 0.05 0.05 0.45 0.05 0.45 0.45 0.05 0.45\n"
            f"{p2} 0.55 0.55 0.95 0.55 0.95 0.95 0.55 0.95\n",
            encoding="utf-8",
        )
        # Alternate between scratch (low), crack (high), and undamaged
        if i % 3 == 0:
            dmg_txt = "0 0.10 0.10 0.25 0.10 0.25 0.25 0.10 0.25\n"
        elif i % 3 == 1:
            dmg_txt = "2 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n"
        else:
            dmg_txt = ""
        (ld_dir / f"car_{i:03d}.txt").write_text(dmg_txt, encoding="utf-8")


def make_synthetic_video(path: Path, n_frames: int = 12):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 64))
    assert writer.isOpened(), f"Failed to create synthetic video at {path}"
    rng = np.random.default_rng(7)
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    writer.release()


def main():
    test_taxonomy()
    proj = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ds = tmp / "syndcar_mini"
        make_synthetic_yolo_dataset(ds, n_images=12)

        csv_path = tmp / "labels.csv"
        subprocess.run(
            [
                sys.executable, str(proj / "build_labels.py"), "yolo",
                "--root", str(ds),
                "--parts-yaml", str(ds / "parts.yaml"),
                "--damage-yaml", str(ds / "damage.yaml"),
                "--source", "smoke",
                "--out", str(csv_path),
            ],
            check=True,
        )
        df = pd.read_csv(csv_path)
        assert len(df) == 12, f"Expected 12 rows, got {len(df)}"
        assert list(df.columns) == ["path", "source"] + PARTS
        assert (df[PARTS].to_numpy() > 0).any(), "Expected some damaged labels in synthetic dataset"
        print("[ok] build_labels.py (yolo mode)")

        run_dir = tmp / "run"
        subprocess.run(
            [
                sys.executable, str(proj / "train.py"),
                "--csv", str(csv_path),
                "--arch", "resnet18",
                "--size", "64",
                "--epochs", "2",
                "--bs", "4",
                "--workers", "0",
                "--no-pretrained",
                "--out", str(run_dir),
            ],
            check=True,
        )
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "meta.json").exists()
        metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
        assert "presence_macro_F1" in metrics and "parts" in metrics
        print("[ok] train.py")

        vid_path = tmp / "demo.avi"
        out_json = tmp / "video_out.json"
        make_synthetic_video(vid_path)
        subprocess.run(
            [
                sys.executable, str(proj / "predict_video.py"),
                "--video", str(vid_path),
                "--run", str(run_dir),
                "--every-sec", "0.2",
                "--out", str(out_json),
            ],
            check=True,
        )
        vres = json.loads(out_json.read_text(encoding="utf-8"))
        assert vres["n_frames"] > 0
        for p in PARTS:
            assert p in vres
            if not vres[p]["damaged"]:
                assert vres[p]["severity"] == "none"
        print("[ok] predict_video.py")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
