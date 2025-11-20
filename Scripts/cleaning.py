import argparse 
import os
from pathlib import Path
import shutil
import json
from collections import defaultdict
from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_SPLITS = {"train", "test"}

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
def faceValid(pillow_image, min_face=40, scaleFactor=1.1, minNeighbors=3): #using openCV's haarcascade to detect faces
    global cv2 
    if cv2 is None:
        import cv2 as _cv2 
        globals()["cv2"] = _cv2

    import numpy as np
    arr = np.array(pillow_image.convert("L")) #convert to grayscale
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(arr, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(min_face, min_face))
    return len(faces) > 0
    
#function to check if image is duplicate using perceptual hashing
def isDuplicate(pillow_image):
    global imagehash
    if imagehash is None:
        import imagehash as _imagehash
        globals()["imagehash"] = _imagehash
    return imagehash.phash(pillow_image)

#function to check image contrast quality
def checkContrast(pillow_image, min_std=15):
    import numpy as np
    arr = np.array(pillow_image.convert("L"))
    return np.std(arr) >= min_std

#function to validate face aspect ratio and size
def checkFaceQuality(pillow_image, min_face_size=30, min_ratio=0.7, max_ratio=1.3, 
                     scaleFactor=1.1, minNeighbors=3):
    global cv2
    if cv2 is None:
        import cv2 as _cv2
        globals()["cv2"] = _cv2
    
    import numpy as np
    arr = np.array(pillow_image.convert("L"))
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(arr, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    if len(faces) == 0:
        return False, "no_face"
    
    # Find largest face
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face
    
    # Check face size
    if min(w, h) < min_face_size:
        return False, "face_too_small"
    
    # Check aspect ratio (faces should be roughly square)
    ratio = w / h if h > 0 else 0
    if ratio < min_ratio or ratio > max_ratio:
        return False, "face_distorted"
    
    # Check for multiple faces(in case for group fotos)
    if len(faces) > 1:
        return True, "multiple_faces"  # Still valid, but flagged
    
    return True, "valid_face"

def keepImage(img_path, min_width, min_height, blur_threshold = None, require_face = False, dedupe_set = None, dedupe_distance = 3,
              min_face=40, face_scale=1.1, face_neighbors=3, check_contrast=False, min_contrast=15,
              check_face_quality=False, min_face_size=30, face_aspect_min=0.7, face_aspect_max=1.3):
    try: 
        im = checkOpen(img_path)
    except (UnidentifiedImageError, OSError): 
        return False, "corrupted", None
    
    if im.width < min_width or im.height < min_height:
        return False, "too_small", None
    
    # Enhanced contrast check
    if check_contrast:
        try:
            if not checkContrast(im, min_std=min_contrast):
                return False, "low_contrast", None
        except Exception:
            return False, "contrast_error", None
    
    if blur_threshold is not None: 
        try: 
            if blurry(im, blur_threshold):
                return False, "blurry", None
        except Exception:
            return False, "blurry_error", None
    
    # Enhanced face quality check
    if check_face_quality:
        try:
            valid, reason = checkFaceQuality(im, min_face_size=min_face_size, 
                                            min_ratio=face_aspect_min, max_ratio=face_aspect_max,
                                            scaleFactor=face_scale, minNeighbors=face_neighbors)
            if not valid:
                return False, reason, None
        except Exception:
            return False, "face_quality_error", None
    elif require_face:
        try:
            if not faceValid(im, min_face=min_face, scaleFactor=face_scale, minNeighbors=face_neighbors):
                return False, "no_face", None
        except Exception:
            return False, "face_detection_error", None
        
    if dedupe_set is not None:
        try: 
            h = isDuplicate(im)
            for existing_hash in dedupe_set:
                if h - existing_hash <= dedupe_distance:
                    return False, "duplicate", None
            dedupe_set.add(h)
        except Exception:
            return False, "dedupe_error", None
    

    return True, "kept", im


def copy_keep(src, dst, img=None, target_size=None):
    """Copy image to destination, optionally resizing it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    if target_size is not None and img is not None:
        # Resize and save the image
        resized = img.resize(target_size, Image.LANCZOS)
        resized.save(dst)
    else:
        # Just copy the original file
        shutil.copy2(src, dst)

def main():
    print("Starting cleaning process...")
    parser = argparse.ArgumentParser(description="Clean FER2013 images and copy to data/cleaned/")
    parser.add_argument("--raw-dir", default="data/raw", type=str, help="Root with train/ and test/ folders")
    parser.add_argument("--out-dir", default="data/cleaned", type=str, help="Destination for cleaned dataset")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Which splits to process")
    parser.add_argument("--min-width", type=int, default=32, help="Minimum width to keep")
    parser.add_argument("--min-height", type=int, default=32, help="Minimum height to keep")

    # still check on these
    parser.add_argument("--blur-threshold", type=float, default=None,
                        help="If set, drop images with Laplacian variance below this (requires opencv-python). Example: 80.0")
    parser.add_argument("--require-face", action="store_true",
                        help="If set, keep only images with a detectable frontal face (requires opencv-python).")
    parser.add_argument("--min-face", type=int, default=40,
                        help="Minimum face size in pixels passed to the detector (smaller = more sensitive).")
    parser.add_argument("--face-scale", type=float, default=1.1,
                        help="Scale factor for the face detector (higher = faster but coarser).")
    parser.add_argument("--face-neighbors", type=int, default=3,
                        help="MinNeighbors parameter for the face detector (lower = more detections, more false positives).")
    parser.add_argument("--dedupe", action="store_true",
                        help="If set, drop near-duplicates using perceptual hash (requires imagehash).")
    parser.add_argument("--dedupe-distance", type=int, default=3,
                        help="Max Hamming distance for near-duplicate detection (lower = stricter).")
    
    # Enhanced quality filters
    parser.add_argument("--check-contrast", action="store_true",
                        help="If set, reject images with poor contrast (too dark or washed out).")
    parser.add_argument("--min-contrast", type=float, default=15.0,
                        help="Minimum standard deviation of pixel values for contrast check (default: 15).")
    parser.add_argument("--check-face-quality", action="store_true",
                        help="If set, validate face size and aspect ratio (stricter than --require-face).")
    parser.add_argument("--min-face-size", type=int, default=30,
                        help="Minimum face dimension in pixels for quality check (default: 30).")
    parser.add_argument("--face-aspect-min", type=float, default=0.7,
                        help="Minimum face aspect ratio (width/height) to avoid distorted faces (default: 0.7).")
    parser.add_argument("--face-aspect-max", type=float, default=1.3,
                        help="Maximum face aspect ratio (width/height) to avoid distorted faces (default: 1.3).")
    
    # Image resizing/standardization
    parser.add_argument("--resize", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                        help="Resize all images to this size (width height). Example: --resize 48 48 or --resize 128 128")

    args = parser.parse_args()
    print("Arguments parsed successfully")

    if args.require_face:
        print(f"Face detection parameters: min_face={args.min_face}, scale={args.face_scale}, neighbors={args.face_neighbors}")
    
    if args.resize:
        print(f"Images will be resized to: {args.resize[0]}x{args.resize[1]}")

    raw_root = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    print(f"Input directory: {raw_root}")
    print(f"Output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    # Stats
    kept = defaultdict(int)
    dropped = defaultdict(lambda: defaultdict(int)) # reason -> class -> count
    summary = {"splits": {}, "params": vars(args)}

    for split in args.splits:
        split_src = raw_root / split
        split_dst = out_root / split

        if not split_src.exists():
            print(f"[!] Missing split folder: {split_src} (skipping)")
            continue

        # one deupe set per class to avoid cross-class suppression
        dedupe_sets = defaultdict(set) if args.dedupe else None

        classes = [d for d in split_src.iterdir() if d.is_dir()]
        split_counts = {"classes": {}, "total_kept": 0, "total_dropped": 0}

        print(f"\nProcessing {split} split...")
        for cls_dir in classes:
            cls_name = cls_dir.name
            images = [p for p in cls_dir.rglob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS]
            print(f"\nProcessing {cls_name} class ({len(images)} images)...")

            cls_kept = 0
            for i, img in enumerate(images, 1):
                if i % 100 == 0:
                    print(f"Progress: {i}/{len(images)} images processed", end="\r")
                keep, reason, pil_img = keepImage(
                    str(img),
                    min_width=args.min_width,
                    min_height=args.min_height,
                    blur_threshold=args.blur_threshold,
                    require_face=args.require_face,
                    dedupe_set=dedupe_sets[cls_name] if dedupe_sets is not None else None,
                    dedupe_distance=args.dedupe_distance,
                    min_face=args.min_face,
                    face_scale=args.face_scale,
                    face_neighbors=args.face_neighbors,
                    check_contrast=args.check_contrast,
                    min_contrast=args.min_contrast,
                    check_face_quality=args.check_face_quality,
                    min_face_size=args.min_face_size,
                    face_aspect_min=args.face_aspect_min,
                    face_aspect_max=args.face_aspect_max
                )
                if keep:
                    dst = split_dst / cls_name / img.name
                    target_size = tuple(args.resize) if args.resize else None
                    copy_keep(img, dst, img=pil_img, target_size=target_size)
                    kept[(split, cls_name)] += 1
                    cls_kept += 1
                else:
                    dropped[reason][cls_name] += 1

            split_counts["classes"][cls_name] = {
                "kept": cls_kept,
                **{f"dropped_{k}": dropped[k].get(cls_name, 0) for k in dropped.keys()}
            }
            split_counts["total_kept"] += cls_kept

        #sum split drops
        split_drops = 0
        for reason in dropped:
            split_drops += sum(v for c, v in dropped[reason].items() if c in [d.name for d in classes])
        split_counts["total_dropped"] = split_drops

        summary["splits"][split] = split_counts

    # Save summary JSON
    (out_root / "clean_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n Cleaning complete. Summary saved to:", out_root / "clean_summary.json")

    # print to terminal
    #for split, info in summary["splits"].items():
     #   print(f"\n== {split} ==")
      #  print(f" kept: {info['total_kept']:,} | dropped: {info['total_dropped']:,}")
       # for cls, stats in sorted(info["classes"].items()):
        #    dropped_breakdown = " ".join([f"{k}:{v}" for k, v in stats.items() if k.startswith("dropped_") and v])
         #   print(f"  - {cls:10s} kept:{stats['kept']:5d} {dropped_breakdown}")

if __name__ == '__main__':
    main()