import argparse
import random
from pathlib import Path
from shutil import copy2, move
import csv

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_images_per_class(src_dir: Path):
    data = {}

    #first check if images are directly under class folders
    immediate = [d for d in src_dir.iterdir() if d.is_dir()]
    immediate.sort(key=lambda p: p.name)
    found_direct = False
    for d in immediate:
        imgs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        if imgs:
            found_direct = True
            data[d.name] = imgs

    if found_direct:
        return data

    #otherwise, look for images under class subfolders
    for split_dir in immediate:
        for c in [p for p in split_dir.iterdir() if p.is_dir()]:
            imgs = [p for p in c.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
            if not imgs:
                continue
            data.setdefault(c.name, []).extend(imgs)

    #sort lists 
    for k in data:
        data[k].sort(key=lambda p: p.name)

    return data

def ensure_empty_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def stratified_split(items, train_ratio, val_ratio, test_ratio, seed):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

def write_manifest(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "filename", "src_path", "dst_path"])
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets.")
    parser.add_argument("--src", default="data/cleaned", help="Folder with cleaned images by class.")
    parser.add_argument("--dst", default="data/processed", help="Output folder for processed data.")
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--move", action="store_true", help="Move instead of copy files.")
    parser.add_argument("--manifest", default="data/meta/processed_split_manifest.csv", help="Path for CSV manifest.")
    args = parser.parse_args()

    # verify sum of ratios is =  1.0
    total_ratio = args.train + args.val + args.test
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0 (got {total_ratio})")

    src = Path(args.src)
    dst = Path(args.dst)
    splits = {"train": dst / "train", "val": dst / "val", "test": dst / "test"}
    for s in splits.values():
        ensure_empty_dir(s)

    data = collect_images_per_class(src)
    if not data:
        raise SystemExit(f"No class folders found in {src}")

    op = move if args.move else copy2
    rows = []

    print(f"\nSplitting dataset from {src} → {dst} (move={args.move})\n")
    for cls, imgs in data.items():
        train_imgs, val_imgs, test_imgs = stratified_split(imgs, args.train, args.val, args.test, args.seed)

        for split_name, items in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            out_dir = splits[split_name] / cls
            ensure_empty_dir(out_dir)
            for p in items:
                dst_path = out_dir / p.name
                op(str(p), str(dst_path))
                rows.append([split_name, cls, p.name, str(p), str(dst_path)])

        print(f"- {cls}: {len(imgs)} total → train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    write_manifest(rows, Path(args.manifest))
   # print(f"\n Done. Manifest saved to: {args.manifest}")

if __name__ == "__main__":
    main()
