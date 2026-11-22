"""Create labels.csv (image, source, front..other in {-1,0,1,2}) from segmentation-style sources.

Per-part severity = f(damage type, damage-area / part-area). THESE RULES ARE HEURISTICS:
validate on a hand-labelled sample (>=300 images) and report agreement in the paper.

Mode A  (CarDD etc.: damage COCO, no parts)  -> parts come from a part-segmentation model
   python build_labels.py coco --images CarDD/images --ann CarDD/instances_train.json \
          --part-model runs/segment/parts/weights/best.pt --source cardd --out labels_cardd.csv
Mode B  (SYNDCAR-style YOLO txt for both parts and damage)
   python build_labels.py yolo --root SYNDCAR --parts-yaml data_parts.yaml --damage-yaml data_damage.yaml \
          --source syndcar --out labels_syndcar.csv
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
from taxonomy import PARTS, map_part, unmapped_names, UNKNOWN

HIGH_TYPES = {"crack", "glass shatter", "lamp broken", "tire flat"}
CFG = dict(min_dmg_frac=0.20,  # damage counts for a part if >=20% of it lies on that part
           dent_high=0.05,     # dent covering >5% of the part -> high
           any_high=0.25)      # any damage covering >25% of the part -> high

def _t(n): return n.lower().replace("_", " ").replace("-", " ")

def label_image(part_masks, damages, cfg=CFG):
    """part_masks: {canonical_part: bool HxW}; damages: [(type_name, bool HxW)] -> {part: 0/1/2/-1}"""
    out = {p: 0 for p in PARTS}
    unassigned = False
    for dtype, dm in damages:
        da = int(dm.sum())
        if da == 0:
            continue
        inter = {p: int((dm & m).sum()) for p, m in part_masks.items()}
        hit = [p for p, i in inter.items() if i / da >= cfg["min_dmg_frac"]]
        if not hit:
            best = max(inter, key=inter.get) if inter else None
            hit = [best] if best and inter[best] > 0 else []
        if not hit:
            unassigned = True
            continue
        for p in hit:
            r = inter[p] / max(int(part_masks[p].sum()), 1)
            t = _t(dtype)
            high = t in HIGH_TYPES or r > cfg["any_high"] or (t == "dent" and r > cfg["dent_high"])
            out[p] = max(out[p], 2 if high else 1)
    if unassigned:  # damage we could not place -> don't claim "no damage" for the other parts
        out = {p: (v if v > 0 else UNKNOWN) for p, v in out.items()}
    return out

def poly_to_mask(pts, h, w):
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or len(arr) < 3 or arr.shape[1] != 2:
        return np.zeros((h, w), bool)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(arr).astype(np.int32)], 1)
    return m.astype(bool)

def union_by_part(items, h, w):
    res = {}
    for name, mask in items:
        p = map_part(name)
        res[p] = res.get(p, np.zeros((h, w), bool)) | mask
    return res

def read_yolo(txt, names, h, w):
    items = []
    p = Path(txt)
    if not p.exists():
        return items
    for line in p.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c, nums = int(v[0]), list(map(float, v[1:]))
        if len(nums) == 4:   # box (coarse!)
            cx, cy, bw, bh = nums
            x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            pts = (np.array(nums).reshape(-1, 2) * [w, h]).tolist()
        else:
            continue
        items.append((names.get(c, str(c)), poly_to_mask(pts, h, w)))
    return items

def load_names(yaml_path):
    import yaml
    with open(yaml_path, encoding="utf-8") as f:
        n = yaml.safe_load(f)["names"]
    if isinstance(n, dict):
        return {int(k): str(v) for k, v in n.items()}
    return {i: str(v) for i, v in enumerate(n)}

def resolve_yolo_label(root, subdir, img, images_root):
    rel = root / subdir / img.relative_to(images_root).with_suffix(".txt")
    if rel.exists():
        return rel
    return root / subdir / f"{img.stem}.txt"

def coco_ann_to_mask(coco, an, h, w):
    seg = an.get("segmentation")
    if seg:
        return coco.annToMask(an).astype(bool)
    bbox = an.get("bbox")
    if bbox and len(bbox) == 4:
        x, y, bw, bh = bbox
        return poly_to_mask([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], h, w)
    return np.zeros((h, w), bool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["coco", "yolo"])
    ap.add_argument("--images"); ap.add_argument("--ann"); ap.add_argument("--part-model")
    ap.add_argument("--root"); ap.add_argument("--parts-yaml"); ap.add_argument("--damage-yaml")
    ap.add_argument("--source", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    if a.mode == "coco":
        if not (a.images and a.ann and a.part_model):
            ap.error("coco mode requires --images, --ann, and --part-model")
        from pycocotools.coco import COCO
        from ultralytics import YOLO
        coco, model = COCO(a.ann), YOLO(a.part_model)
        cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
        for iid, info in coco.imgs.items():
            img_path = Path(a.images) / info["file_name"]
            if not img_path.exists():
                print(f"warning: missing image {img_path}, skipping")
                continue
            h, w = info["height"], info["width"]
            dmg = [(cats[an["category_id"]], coco_ann_to_mask(coco, an, h, w))
                   for an in coco.loadAnns(coco.getAnnIds(imgIds=iid))]
            r = model.predict(str(img_path), verbose=False)[0]
            parts = []
            if r.masks is not None and r.boxes is not None:
                for cls, xy in zip(r.boxes.cls.tolist(), r.masks.xy):
                    if len(xy) >= 3:
                        parts.append((model.names[int(cls)], poly_to_mask(xy, h, w)))
            lab = label_image(union_by_part(parts, h, w), dmg)
            rows.append([str(img_path), a.source] + [lab[p] for p in PARTS])
    else:
        if not (a.root and a.parts_yaml and a.damage_yaml):
            ap.error("yolo mode requires --root, --parts-yaml, and --damage-yaml")
        root = Path(a.root)
        images_root = root / "images"
        pn, dn = load_names(a.parts_yaml), load_names(a.damage_yaml)
        for img in sorted(images_root.rglob("*.*")):
            im = cv2.imread(str(img))
            if im is None:
                continue
            h, w = im.shape[:2]
            parts = read_yolo(resolve_yolo_label(root, "labels_parts", img, images_root), pn, h, w)
            dmg = read_yolo(resolve_yolo_label(root, "labels_damage", img, images_root), dn, h, w)
            lab = label_image(union_by_part(parts, h, w), dmg)
            rows.append([str(img), a.source] + [lab[p] for p in PARTS])
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "source"] + PARTS)
        wr.writerows(rows)
    print(f"{len(rows)} rows -> {out_path}; unmapped part names: {unmapped_names()}")

if __name__ == "__main__":
    main()
