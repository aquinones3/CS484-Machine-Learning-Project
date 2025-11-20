import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from collections import Counter

def get_data(dir, batch_size=32, use_weighted_sampling=True):
    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "val")
    test_dir = os.path.join(dir, "test")

    # training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Validation/test transforms without augmentation
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_data = datasets.ImageFolder(train_dir, train_transform)
    val_data = datasets.ImageFolder(val_dir, val_test_transform)
    test_data = datasets.ImageFolder(test_dir, val_test_transform)

    use_cuda = torch.cuda.is_available()
    
    # handle class imbalance with weighted sampling
    if use_weighted_sampling:
        # Calculate class weights
        class_counts = Counter(train_data.targets)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[target] for target in train_data.targets]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train = DataLoader(train_data, batch_size, sampler=sampler, pin_memory=use_cuda)
        print(f"\nClass distribution: {dict(class_counts)}")
        print("Using weighted sampling to balance classes")
    else:
        train = DataLoader(train_data, batch_size, shuffle=True, pin_memory=use_cuda)
    
    val = DataLoader(val_data, batch_size, pin_memory=use_cuda)
    test = DataLoader(test_data, batch_size, pin_memory=use_cuda)

    return train, val, test, train_data.classes, class_counts if use_weighted_sampling else None

def build_model(num_classes):
    model = nn.Sequential(
        nn.Conv2d(1, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, 1, 1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, 3, 1, 1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(256, 512, 3, 1, 1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(512 * 3 * 3, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes)
    )
    return model

def train(model, data, loss_func, optimizer, device):
    model.train()

    for images, labels in data:
        image = images.to(device)
        label = labels.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        
    return loss

def eval_model(model, loader, loss_func, device):
    model.eval()
    
    with torch.no_grad():
        for imgs, labels in loader:
            image = imgs.to(device)
            label = labels.to(device)
            outputs = model(image)
            loss = loss_func(outputs, label)
            
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed")
    parser.add_argument("--use-weighted-sampling", action="store_true", default=True,
                        help="Use weighted sampling to balance classes (default: True)")
    parser.add_argument("--use-weighted-loss", action="store_true",
                        help="Use weighted loss function to handle class imbalance")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    training, val, test, classes, class_counts = get_data(args.dir, use_weighted_sampling=args.use_weighted_sampling)
    
    model = build_model(num_classes=len(classes))
    model = model.to(device)

    # [Optionally] use weighted loss for class imbalance
    if args.use_weighted_loss and class_counts:
        total = sum(class_counts.values())
        class_weights = torch.tensor([total / class_counts[i] for i in range(len(classes))], dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(classes)  # Normalize
        loss_func = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(f"Using weighted loss with weights: {class_weights.tolist()}")
    else:
        loss_func = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters())

    train_loss = train(model, training, loss_func, optimizer, device)

    val_loss = eval_model(model, val, loss_func, device)

    print("Training Loss:", train_loss,"Validation Loss:", val_loss)

    test_loss = eval_model(model, test, loss_func, device)
    print("Test Loss:", test_loss)

if __name__ == "__main__":
    main()