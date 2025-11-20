import argparse
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_stats(data_dir, batch_size=64):
    """
    Calculating mean and std of the dataset.
    
    How it works:
    1. Load all training images
    2. Convert to tensors (pixel values 0-1)
    3. Calculate mean across all pixels in all images
    4. Calculate std across all pixels in all images
    """
    
    # transform to load images as grayscale tensors
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),  # use target size
        transforms.ToTensor()  # Converts to [0, 1] range
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    
    # Variables to accumulate mean and std
    mean = 0.0
    std = 0.0
    total_images = 0
    
    for images, _ in tqdm(loader, desc="Processing batches"):
        batch_samples = images.size(0)  # # of images in this batch
        
        # reshape: [batch, channels, height, width] -> [batch, channels, height*width]
        images = images.view(batch_samples, images.size(1), -1)
        
        # calculate mean and std for this batch
        mean += images.mean(2).sum(0)  # mean across height*width, sum across batch
        std += images.std(2).sum(0)    # std across height*width, sum across batch
        
        total_images += batch_samples
    
    #divide by total # of images to get final mean and std
    mean /= total_images
    std /= total_images
    
    return mean.item(), std.item()


def main():
    parser = argparse.ArgumentParser(description="Calculate dataset normalization statistics")
    parser.add_argument("--dir", default="data/processed/train", 
                        help="Path to training data directory")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing (default: 64)")
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory {args.dir} does not exist!")
        print("Make sure to run this on your processed training data.")
        return
    
    mean, std = calculate_stats(args.dir, args.batch_size)
    
    print("\n" + "="*60)
    print("DATASET NORMALIZATION STATISTICS")
    print("="*60)
    print(f"Mean: {mean:.4f}")
    print(f"Std:  {std:.4f}")
    print("="*60)
    
    print(f"\nUpdate train_model.py with:")
    print(f"transforms.Normalize(mean=[{mean:.4f}], std=[{std:.4f}])")


if __name__ == "__main__":
    main()
