import os
import csv
import argparse
import torch
from torch import nn
import matplotlib.pyplot as plt


from train_model import get_data, build_model

def train_one_epoch(model, data, loss_func, optimizer, device):
    """Uses your variable names: data, loss_func, optimizer, device."""
    model.train()
    total, correct, running = 0, 0, 0.0
    for images, labels in data:
        image = images.to(device)
        label = labels.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        running += loss.item() * image.size(0)
        pred = output.argmax(1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    avg_loss = running / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

@torch.no_grad()
def eval_epoch(model, loader, loss_func, device):
    """Keeps your naming: loader, loss_func, device."""
    model.eval()
    total, correct, running = 0, 0, 0.0
    for imgs, labels in loader:
        image = imgs.to(device)
        label = labels.to(device)
        outputs = model(image)
        loss = loss_func(outputs, label)

        running += loss.item() * image.size(0)
        pred = outputs.argmax(1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    avg_loss = running / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed")  
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)  
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    data = get_data(args.dir, batch_size=args.batch_size)
    training, val, test, classes = data[:4]   # ignore any extra values


    model = build_model(num_classes=len(classes)).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    rows = []
    results_dir = "results"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, training, loss_func, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val,     loss_func, device)

        train_losses.append(train_loss); train_accs.append(train_acc)
        val_losses.append(val_loss);     val_accs.append(val_acc)
        rows.append([epoch, train_loss, val_loss, train_acc, val_acc])

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    
    test_loss, test_acc = eval_epoch(model, test, loss_func, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2%}")

    os.makedirs(results_dir, exist_ok=True)
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
