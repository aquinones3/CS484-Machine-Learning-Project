import os
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def get_data(dir, batch_size=32):
    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "val")
    test_dir = os.path.join(dir, "test")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(train_dir, transform)
    val_data = datasets.ImageFolder(val_dir, transform)
    test_data = datasets.ImageFolder(test_dir, transform)

    use_cuda = torch.cuda.is_available()
    train = DataLoader(train_data, batch_size, shuffle=True, pin_memory=use_cuda)
    val = DataLoader(val_data, batch_size, pin_memory=use_cuda)
    test = DataLoader(test_data, batch_size, pin_memory=use_cuda)

    return train, val, test, train_data.classes

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
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    training, val, test, classes = get_data(args.dir)
    
    model = build_model(num_classes=len(classes))
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    train_loss = train(model, training, loss_func, optimizer, device)

    val_loss = eval_model(model, val, loss_func, device)

    print("Training Loss:", train_loss,"Validation Loss:", val_loss)

    test_loss = eval_model(model, test, loss_func, device)
    print("Test Loss:", test_loss)

if __name__ == "__main__":
    main()