# Scripts/data_analysis.py
import os, csv, itertools, random, warnings
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # parent of Scripts/

# ---------- Optional deps ----------
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from sklearn.metrics import classification_report
    HAS_SK = True
except Exception:
    HAS_SK = False

# ---- import your helpers (unchanged) ----
from train_model import get_data, build_model

# -------------------------- utils --------------------------
def _pil_open(path):
    from PIL import Image
    return Image.open(path).convert("RGB")

def savefig_safe(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p))
    plt.close()

def conv2(img, k):
    """tiny same-conv for Sobel fallback (no padding at border)."""
    h, w = img.shape
    kh, kw = k.shape
    pad_h, pad_w = kh//2, kw//2
    padded = np.pad(img, ((pad_h,pad_h),(pad_w,pad_w)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i,j] = np.sum(region * k)
    return out

# -----------------------------------
# Count files in each split / class
# -----------------------------------
def count_split_and_classes(root_dir):
    root = Path(root_dir)
    splits = []
    per_class = {}  # {split: {class_name: count}}
    total_by_split = {}  # {split: total_count}

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            warnings.warn(f"Missing split: {split_dir}")
            continue

        cls_counts = {}
        total = 0
        for cls_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
            cnt = sum(1 for _ in cls_dir.rglob("*") if _.is_file())
            cls_counts[cls_dir.name] = cnt
            total += cnt
        per_class[split] = cls_counts
        total_by_split[split] = total
        splits.append(split)

    return splits, total_by_split, per_class

# -----------------------------------
# Figures: dataset structure
# -----------------------------------
def plot_split_distribution(total_by_split, out_path):
    names  = list(total_by_split.keys())          # ["train","val","test"]
    values = [total_by_split[k] for k in names]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    plt.figure(figsize=(8,5))
    plt.title("Dataset Distribution")
    plt.bar(names, values, color=colors)
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of images")
    for x, v in zip(range(len(values)), values):
        plt.text(x, v, f"{v}", ha="center", va="bottom")
    plt.tight_layout()
    savefig_safe(out_path)

def plot_per_class_distribution(per_class, out_path):
    splits = list(per_class.keys())
    class_names = sorted(set(itertools.chain.from_iterable(per_class[s].keys() for s in splits)))
    bottoms = np.zeros(len(class_names))
    plt.figure(figsize=(10, 6))
    for split in splits:
        counts = np.array([per_class[split].get(c, 0) for c in class_names])
        plt.bar(class_names, counts, bottom=bottoms, label=split)
        bottoms += counts
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Images per class")
    plt.title("Per-class distribution by split")
    plt.legend()
    plt.tight_layout()
    savefig_safe(out_path)

def save_sample_and_edges(root_dir, out_dir):
    """Pick one random image from train and save: original, grayscale, edges, overlay."""
    rng = random.Random(42)
    train_root = Path(root_dir) / "train"
    files = [p for p in train_root.rglob("*") if p.is_file()]
    if not files:
        warnings.warn("No images found in train/ to sample.")
        return
    img_path = rng.choice(files)

    img_rgb = np.array(_pil_open(img_path))  # HxWx3
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if HAS_CV2 else np.mean(img_rgb, axis=2).astype(np.uint8)

    if HAS_CV2:
        edges = cv2.Canny(gray, 100, 200)
        overlay = img_rgb.copy()
        overlay[edges > 0] = [0, 255, 0]  # green edges
    else:
        kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
        ky = kx.T
        gx = conv2(gray, kx)
        gy = conv2(gray, ky)
        mag = np.sqrt(gx**2 + gy**2)
        edges = (255 * (mag / (mag.max() + 1e-8))).astype(np.uint8)
        overlay = np.dstack([gray, gray, gray])
        overlay[edges > (edges.mean()+edges.std())] = [0, 255, 0]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_rgb); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(gray, cmap="gray"); axs[1].set_title("Grayscale"); axs[1].axis("off")
    axs[2].imshow(overlay); axs[2].set_title("Edges overlay"); axs[2].axis("off")
    fig.suptitle(f"Sample from train: {img_path.parent.name}")
    plt.tight_layout()
    savefig_safe(out_dir / "sample_and_edges.png")

# -----------------------------------
# Confusion matrix on test
# -----------------------------------
@torch.no_grad()
def eval_confusion_matrix_counts(model, loader, classes, device, out_path):
    """
    Confusion matrix with raw COUNTS (not normalized), styled like the example.
    Saves a PNG to `out_path` and returns (cm, y_true, y_pred).
    """
    import numpy as np

    try:
        import seaborn as sns
        use_sns = True
    except Exception:
        use_sns = False

    model.eval()
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    all_true, all_pred = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        for t, p in zip(y.view(-1), pred.view(-1)):
            ti, pi = int(t), int(p)
            cm[ti, pi] += 1
            all_true.append(ti)
            all_pred.append(pi)

    tick_labels = [c.capitalize() for c in classes]

    plt.figure(figsize=(8, 6))
    if use_sns:
        import seaborn as sns
        sns.heatmap(
            cm,
            annot=True, fmt="d", cmap="Blues",
            xticklabels=tick_labels, yticklabels=tick_labels,
            cbar=True
        )
    else:
        plt.imshow(cm, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(n), tick_labels, rotation=45, ha="right")
        plt.yticks(range(n), tick_labels)
        for i in range(n):
            for j in range(n):
                plt.text(j, i, str(cm[i, j]),
                         ha="center", va="center", color="black", fontsize=8)

    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    savefig_safe(out_path)

    return cm, np.array(all_true, dtype=int), np.array(all_pred, dtype=int)

# ---------- 1: normalized confusion matrix ----------
def plot_normalized_confusion(cm, classes, out_path):
    n = len(classes)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    tick_labels = [c.capitalize() for c in classes]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(range(n), tick_labels, rotation=45, ha="right")
    plt.yticks(range(n), tick_labels)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{cm_norm[i, j]*100:.0f}%",
                     ha="center", va="center", color="black", fontsize=8)
    plt.title("Confusion Matrix (Row-Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    savefig_safe(out_path)

# ---------- 2: per-class accuracy bar chart ----------
def plot_per_class_accuracy(cm, classes, out_path):
    correct = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    acc = correct / np.maximum(support, 1)
    idx = np.argsort(-acc)  # sort by accuracy high→low

    names_sorted = [classes[i].capitalize() for i in idx]
    acc_sorted = acc[idx]
    sup_sorted = support[idx]

    plt.figure(figsize=(10, 5))
    plt.bar(names_sorted, acc_sorted)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Per-class Accuracy")
    plt.title("Per-class Accuracy and Support")
    for i, (a, s) in enumerate(zip(acc_sorted, sup_sorted)):
        plt.text(i, a + 0.02, f"n={int(s)}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    savefig_safe(out_path)

# ---------- 3: top-k (top-1, top-3) accuracy ----------
@torch.no_grad()
def compute_topk_accuracies(model, loader, device, num_classes, ks=(1,3)):
    model.eval()
    correct = {k: 0 for k in ks}
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        # sort descending on logits
        topk = torch.topk(logits, k=max(ks), dim=1).indices  # [B, max_k]
        total += y.size(0)
        for k in ks:
            preds_k = topk[:, :k]  # [B, k]
            match = (preds_k == y.unsqueeze(1)).any(dim=1)
            correct[k] += match.sum().item()
    return {k: (correct[k] / max(total,1)) for k in ks}

def write_topk_file(topk_dict, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in topk_dict.items():
            f.write(f"Top-{k} accuracy: {v:.4f}\n")

# ---------- 4: classification report (precision/recall/F1) ----------
def write_classification_report(y_true, y_pred, classes, out_dir):
    if not HAS_SK:
        print("scikit-learn not installed; skipping classification report.")
        return
    target_names = [c.capitalize() for c in classes]
    rep = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    # TXT (pretty)
    txt_path = Path(out_dir) / "classification_report.txt"
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        ))
    # CSV
    csv_path = Path(out_dir) / "classification_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","precision","recall","f1","support"])
        for label, stats in rep.items():
            if isinstance(stats, dict) and "precision" in stats:
                w.writerow([
                    label,
                    f"{stats['precision']:.4f}",
                    f"{stats['recall']:.4f}",
                    f"{stats['f1-score']:.4f}",
                    int(stats['support'])
                ])

# ---------- 5: most-confused class pairs ----------
def write_most_confused_pairs(cm, classes, out_path, top_n=20):
    rows = []
    n = len(classes)
    for i in range(n):
        for j in range(n):
            if i == j: 
                continue
            count = int(cm[i,j])
            if count > 0:
                rows.append((classes[i], classes[j], count))
    rows.sort(key=lambda x: x[2], reverse=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true","pred","count"])
        w.writerows(rows[:top_n])

# -----------------------------------
# Main
# -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(PROJECT_ROOT / "data" / "processed"),
                    help="Folder with train/ val/ test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)  # 0 is Windows-friendly
    ap.add_argument("--weights", default=str(PROJECT_ROOT / "results" / "emotion_cnn.pt"),
                    help="Optional: path to trained weights for confusion matrix")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "analysis"),
                    help="Where to save figures")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    print("CWD:", os.getcwd())
    print("Script:", __file__)
    print("Using --dir    :", os.path.abspath(args.dir))
    print("Using --weights:", os.path.abspath(args.weights))
    print("Using --out    :", os.path.abspath(args.out))

    # ---- 1) Dataset distribution plots ----
    splits, total_by_split, per_class = count_split_and_classes(args.dir)
    if total_by_split:
        plot_split_distribution(total_by_split, Path(args.out) / "dataset_split_distribution.png")
        plot_per_class_distribution(per_class, Path(args.out) / "per_class_distribution.png")
    save_sample_and_edges(args.dir, args.out)

    # ---- 2) Load data & model for confusion matrix ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training, val, test, classes = get_data(args.dir, batch_size=args.batch_size)
    model = build_model(num_classes=len(classes)).to(device)

    if args.weights and os.path.exists(args.weights):
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded weights: {args.weights}")
    else:
        print("WARNING: weights not found; confusion matrix and extra analyses will be based on an untrained/random model.")

    # ---- 3) Confusion matrix + new analyses ----
    cm, y_true, y_pred = eval_confusion_matrix_counts(
        model, test, classes, device,
        Path(args.out) / "confusion_matrix_counts.png"
    )

    #1: normalized confusion matrix
    plot_normalized_confusion(cm, classes, Path(args.out) / "confusion_matrix_normalized.png")

    #2: per-class accuracy bar chart
    plot_per_class_accuracy(cm, classes, Path(args.out) / "per_class_accuracy.png")

    #3: top-k accuracy (top-1, top-3)
    topk = compute_topk_accuracies(model, test, device, num_classes=len(classes), ks=(1,3))
    write_topk_file(topk, Path(args.out) / "topk_metrics.txt")

    #4: classification report (precision/recall/F1)
    if y_true.size > 0:
        write_classification_report(y_true, y_pred, classes, args.out)

    #5: most-confused class pairs
    write_most_confused_pairs(cm, classes, Path(args.out) / "most_confused_pairs.csv")

    # ---- 4) Summarize counts to CSV ----
    csv_path = Path(args.out) / "counts_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split","class","count"])
        for s in per_class:
            for c, v in per_class[s].items():
                w.writerow([s,c,v])

    print(f"Saved analysis to: {args.out}")
    print(" - dataset_split_distribution.png")
    print(" - per_class_distribution.png")
    print(" - sample_and_edges.png")
    print(" - confusion_matrix_counts.png")
    print(" - confusion_matrix_normalized.png")
    print(" - per_class_accuracy.png")
    print(" - topk_metrics.txt")
    print(" - classification_report.(txt|csv)")
    print(" - most_confused_pairs.csv")
    print(" - counts_summary.csv")

if __name__ == "__main__":
    main()
