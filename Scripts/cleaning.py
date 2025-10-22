import argparse 
import os
from pathlib import Path
import shutil
import json
from collections import defaultdict
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SPLITS = {"train", "train"}

cv2 = None # 
imagehash = None # 


#function to check if image can be opened
def checkOpen(fp):
    with Image.open(fp) as im:
        im.load()
        return im

#function to check if image is blurry by using variance of laplacian 
def blurry(pillow_image, blur_threshold): #in simple terms, sharper images will have more variation in pixel density compared to blurry ones that will have a lower variation
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        globals()["cv2"] = _cv2

    import numpy as np
    arr = np.array(pillow_image.convert("L")) #convert to grayscale 
    fm = cv2.Laplacian(arr, cv2.CV_64F).var() # compute the Laplacian of the image and then return the variance
    return fm < blur_threshold

#function to check if face is deteced in the image
def faceValid(pillow_image, min_face=40): #using openCV's haarcascade to detect faces
    global cv2 
    if cv2 is None:
        import cv2 as _cv2 
        globals()["cv2"] = _cv2

    import numpy as np
    arr = np.array(pillow_image.convert("L")) #convert to grayscale
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(arr, scaleFactor=1.1, minNeighbors=3, minSize=(min_face, min_face))
    return len(faces) > 0
    
#function to check if image is duplicate using perceptual hashing
def isDuplicate(pillow_image):
    global imagehash
    if imagehash is None:
        import imagehash as _imagehash
        globals()["imagehash"] = _imagehash
    return imagehash.phash(pillow_image)

def keepImage(img_path, min_width, min_height, blur_threshold = None, require_face = False, dedupe_set = None, dedupe_distance = 3):
    try: 
        im = checkOpen(img_path)
    except (UnidentifiedImageError, OSError): 
        return False, "corrupted"
    
    if im.width < min_width or im.height < min_height:
        return False, "too_small"
    
    if blur_threshold is not None: 
        try: 
            if blurry(im, blur_threshold):
                return False, "blurry"
        except Exception:
            return False, "blurry_error"
    if require_face:
        try:
            if not faceValid(im):
                return False, "no_face"
        except Exception:
            return False, "face_detection_error"
        
    if dedupe_set is not None:
        try: 
            h = isDuplicate(im)
            for existing_hash in dedupe_set:
                if h - existing_hash <= dedupe_distance:
                    return False, "duplicate"
            dedupe_set.add(h)
        except Exception:
            return False, "dedupe_error"
    

    return True, "kept"


def copy_keep(src,dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src,dst)

def main():
    parser = argparse.ArgumentParser(description="Clean FER2013 images and copy to data/cleaned/")
    parser.add_argument("--raw-dir", default="data/raw", type=str, help="Root with train/ and test/ folders")
    parser.add_argument("--out-dir", default="data/cleaned", type=str, help="Destination for cleaned dataset")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Which splits to process")
    parser.add_argument("--min-width", type=int, default=32, help="Minimum width to keep")
    parser.add_argument("--min-height", type=int, default=32, help="Minimum height to keep")

    # Optional filters
    parser.add_argument("--blur-threshold", type=float, default=None,
                        help="If set, drop images with Laplacian variance below this (requires opencv-python). Example: 80.0")
    parser.add_argument("--require-face", action="store_true",
                        help="If set, keep only images with a detectable frontal face (requires opencv-python).")
    parser.add_argument("--dedupe", action="store_true",
                        help="If set, drop near-duplicates using perceptual hash (requires imagehash).")
    parser.add_argument("--dedupe-distance", type=int, default=3,
                        help="Max Hamming distance for near-duplicate detection (lower = stricter).")

    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Stats
    kept = defaultdict(int)
    dropped = defaultdict(lambda: defaultdict(int))  # dropped[reason][class] += 1
    summary = {"splits": {}, "params": vars(args)}

    for split in args.splits:
        split_src = raw_root / split
        split_dst = out_root / split

        if not split_src.exists():
            print(f"[!] Missing split folder: {split_src} (skipping)")
            continue

        # optional: fresh start for the split
        # shutil.rmtree(split_dst, ignore_errors=True)

        # one dedupe set per class to avoid cross-class suppression
        dedupe_sets = defaultdict(set) if args.dedupe else None

        classes = [d for d in split_src.iterdir() if d.is_dir()]
        split_counts = {"classes": {}, "total_kept": 0, "total_dropped": 0}

        for cls_dir in classes:
            cls_name = cls_dir.name
            images = [p for p in cls_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS]

            cls_kept = 0
            for img in images:
                keep, reason = keepImage(
                    img,
                    min_w=args.min_width,
                    min_h=args.min_height,
                    blur_thresh=args.blur_thresh,
                    require_face=args.require_face,
                    dedupe_set=dedupe_sets[cls_name] if dedupe_sets is not None else None,
                    dedupe_distance=args.dedupe_distance
                )
                if keep:
                    dst = split_dst / cls_name / img.name
                    copy_keep(img, dst)
                    kept[(split, cls_name)] += 1
                    cls_kept += 1
                else:
                    dropped[reason][cls_name] += 1

            split_counts["classes"][cls_name] = {
                "kept": cls_kept,
                **{f"dropped_{k}": dropped[k].get(cls_name, 0) for k in dropped.keys()}
            }
            split_counts["total_kept"] += cls_kept

        # sum per-split drops
        split_drops = 0
        for reason in dropped:
            split_drops += sum(v for c, v in dropped[reason].items() if c in [d.name for d in classes])
        split_counts["total_dropped"] = split_drops

        summary["splits"][split] = split_counts

    # Save summary JSON
    (out_root / "clean_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[✓] Cleaning complete. Summary saved to:", out_root / "clean_summary.json")

    # Quick terminal recap
    for split, info in summary["splits"].items():
        print(f"\n== {split} ==")
        print(f" kept: {info['total_kept']:,} | dropped: {info['total_dropped']:,}")
        for cls, stats in sorted(info["classes"].items()):
            dropped_breakdown = " ".join([f"{k}:{v}" for k, v in stats.items() if k.startswith("dropped_") and v])
            print(f"  - {cls:10s} kept:{stats['kept']:5d} {dropped_breakdown}")