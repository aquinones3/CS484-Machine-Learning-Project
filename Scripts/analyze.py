# analyze.py
import os
import argparse
import csv
import torch
from torch import nn
import matplotlib.pyplot as plt

# import your helpers from train_model.py (no edits needed there)
from train_model import get_data, build_model

@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)

def train_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)

def save_plots(train_losses, val_losses, train_accs, val_accs, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Loss curve
    plt.figure()
    plt.title("Loss per Epoch")
    plt.plot(range(1, len(train_losses)+1), train_losses, label="train")
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"))
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.title("Accuracy per Epoch")
    plt.plot(range(1, len(train_accs)+1), train_accs, label="train")
    plt.plot(range(1, len(val_accs)+1),   val_accs,   label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_curve.png"))
    plt.close()

def write_csv(rows, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","val_loss","train_acc","val_acc"])
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="data/processed", help="Folder containing train/ val/ test/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_model", action="store_true", help="Also save trained weights")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse your data builder (expects train/, val/, test/ under --dir)
    train_loader, val_loader, test_loader, classes, counts = get_data(
        args.dir, batch_size=args.batch_size
    )

    model = build_model(num_classes=len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    results_dir = "results"
    train_losses, val_losses, train_accs, val_accs, rows = [], [], [], [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_fn, opt, device)
        va_loss, va_acc = eval_epoch(model, val_loader,   loss_fn, device)

        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(va_loss);   val_accs.append(va_acc)
        rows.append([epoch, tr_loss, va_loss, tr_acc, va_acc])

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f}")

    # Final test metrics
    test_loss, test_acc = eval_epoch(model, test_loader, loss_fn, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}")

    # Save artifacts
    save_plots(train_losses, val_losses, train_accs, val_accs, results_dir)
    write_csv(rows, os.path.join(results_dir, "metrics.csv"))
    with open(os.path.join(results_dir, "test_metrics.txt"), "w") as f:
        f.write(f"Test Loss: {test_loss:.6f}\nTest Accuracy: {test_acc:.6f}\n")

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(results_dir, "emotion_cnn.pt"))

    print(f"Saved plots to {results_dir}/loss_curve.png and {results_dir}/accuracy_curve.png")
    print(f"Saved CSV to {results_dir}/metrics.csv and summary to {results_dir}/test_metrics.txt")
    if args.save_model:
        print(f"Saved model to {results_dir}/emotion_cnn.pt")

if __name__ == "__main__":
    main()