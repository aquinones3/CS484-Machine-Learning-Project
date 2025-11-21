# Data Preparation Guide

## Overview
The data prep pipeline has 3 main steps:
1. **Cleaning** - Filter and resize images
2. **Splitting** - Divide into train/val/test sets
3. **Calculate Stats** - Get normalization values for training

---

## Step 1: Cleaning (`cleaning.py`)

Cleans raw images by filtering out poor quality images and resizing to a standard size.

### Basic Usage
```bash
python Scripts/cleaning.py --raw-dir data/raw --out-dir data/cleaned_v2 --resize 128 128
```

### Available Flags

#### **Required/Common Flags**
- `--raw-dir` - Input directory with train/ and test/ folders (default: `data/raw`)
- `--out-dir` - Output directory for cleaned images (default: `data/cleaned`)
- `--resize WIDTH HEIGHT` - Resize all images to this size (e.g., `--resize 128 128`)

#### **Size Filtering**
- `--min-width` - Minimum image width in pixels (default: 32)
- `--min-height` - Minimum image height in pixels (default: 32)

#### **Quality Filters**
- `--check-contrast` - Reject images with poor contrast (too dark/bright)
- `--min-contrast` - Minimum contrast threshold (default: 15.0)
- `--blur-threshold` - Drop blurry images below this threshold (e.g., `80.0`)
- `--dedupe` - Remove near duplicate images

#### **Face Detection Filters**
- `--require-face` - Keep only images with detectable faces
- `--check-face-quality` - Stricter face validation (size + aspect ratio)
- `--min-face` - Minimum face size for detection (default: 40)
- `--min-face-size` - Minimum face dimension for quality check (default: 30)
- `--face-aspect-min` - Min face aspect ratio (default: 0.7)
- `--face-aspect-max` - Max face aspect ratio (default: 1.3)
- `--face-scale` - Face detector scale factor (default: 1.1)
- `--face-neighbors` - Face detector sensitivity (default: 3, lower = more sensitive)

### Example Commands

**Basic cleaning with resize:**
```bash
python Scripts/cleaning.py --resize 128 128 --out-dir data/cleaned_v2
```

**With quality filters:**
```bash
python Scripts/cleaning.py --resize 128 128 --check-contrast --blur-threshold 80 --out-dir data/cleaned_v2
```

**Strict quality control:**
```bash
python Scripts/cleaning.py --resize 128 128 --check-contrast --check-face-quality --blur-threshold 100 --dedupe --out-dir data/cleaned_v2
```
---

## Step 2: Splitting (`splitTrain.py`)

Splits cleaned data into train/validation/test sets.

### Basic Usage
```bash
python Scripts/splitTrain.py --src data/cleaned_v2 --dst data/processed_v2 --move
```

### Available Flags

- `--src` - Source directory with class folders
- `--dst` - Destination directory for split data
- `--train` - Training set ratio (default: 0.7 = 70%)
- `--val` - Validation set ratio (default: 0.15 = 15%)
- `--test` - Test set ratio (default: 0.15 = 15%)
- `--move` - Move files instead of copying (saves disk space)
- `--seed` - Random seed for reproducibility (default: 42)
- `--manifest` - Path to save split manifest CSV

### Example Commands

**Standard 70/15/15 split:**
```bash
python Scripts/splitTrain.py --src data/cleaned_v2 --dst data/processed_v2 --move
```

**Custom split ratio (80/10/10):**
```bash
python Scripts/splitTrain.py --src data/cleaned_v2 --dst data/processed_v2 --train 0.8 --val 0.1 --test 0.1
```

---

## Step 3: Calculate Normalization (`calculate_normalization.py`)

Calculates the actual mean and standard deviation of training dataset for proper normalization.

### Basic Usage
```bash
python Scripts/calculate_normalization.py --dir data/processed_v2/train
```

### Available Flags

- `--dir` - Path to training data directory (default: `data/processed/train`)
- `--batch-size` - Batch size for processing (default: 64)

### Example Command
```bash
python Scripts/calculate_normalization.py --dir data/processed_v2/train
```

### Sample Output
```
DATASET NORMALIZATION STATISTICS
============================================================
Mean: 0.5077
Std:  0.2062
============================================================

Update train_model.py with:
transforms.Normalize(mean=[0.5077], std=[0.2062])
```

**Copy these values and use them in training script**

---

## Training Script Updates (`train_model.py`)

### What Was Added

#### 1. **Data Augmentation**
Training images now get random transformations to prevent overfitting:
- Random horizontal flips (50% chance)
- Random rotation (±15 degrees)
- Proper normalization with dataset statistics

#### 2. **Class Imbalance Handling**
Two methods to handle imbalanced classes (e.g., disgust has fewer images than happy):

**Weighted Sampling (Default - Enabled):**
- Automatically oversamples minority classes during training
- Each class gets equal representation per epoch

**Weighted Loss (Optional):**
- Penalizes mistakes on minority classes more heavily
- Use with `--use-weighted-loss` flag

### Training Script Flags

- `--dir` - Path to processed data directory (default: `data/processed`)
- `--use-weighted-sampling` - Balance classes via sampling (default: True)
- `--use-weighted-loss` - Use weighted loss function (default: False)

### Example Training Commands

**Standard training with weighted sampling:**
```bash
python Scripts/train_model.py --dir data/processed_v2
```

**With both weighted sampling and weighted loss:**
```bash
python Scripts/train_model.py --dir data/processed_v2 --use-weighted-loss
```

---

## Complete Pipeline Example


```bash
# Step 1: Clean and resize images
python Scripts/cleaning.py --raw-dir data/raw --out-dir data/cleaned_v2 --resize 128 128 --check-contrast

# Step 2: Split into train/val/test
python Scripts/splitTrain.py --src data/cleaned_v2 --dst data/processed_v2 --move

# Step 3: Calculate normalization statistics
python Scripts/calculate_normalization.py --dir data/processed_v2/train

# Step 4: Update train_model.py with the calculated mean/std values
# (manually edit the Normalize transform with the output values)

# Step 5: Train the model
python Scripts/train_model.py --dir data/processed_v2
```

---

## Troubleshooting

**Issue: Too many images dropped during cleaning**
- Loosen filters: increase `--face-neighbors`, decrease `--blur-threshold`
- Remove strict filters like `--check-face-quality` initially

**Issue: Class imbalance still problematic**
- Enable both `--use-weighted-sampling` and `--use-weighted-loss`