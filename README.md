# Emotion Recoginition using CNNs

This project detects human emotions from facial expressions using Convolutional Neural Networks(CNNS)

## Table of contents
- [Team Members](#team-members)
- [Folder Structure](#folder-structure)
- [Dataset](#dataset)
- [More on Data Prep](#more-on-data-prep)
- [Running Model](#running-model)
- [Model](#model)
- [Analysis](#analysis)

# Team Members 
- Andres Quinones [Data Prep]
- Eric Mattner [Model]
- Name [what you worked on]

## Folder Structure
- data - contains folders to load data in (test/train/validation)
- Scripts - preprocessing and ultility scripts
- gitignore - exluces the large data sets
- [add more if needed]

## Dataset 
We use the [FER2013 Dataset] (https://www.kaggle.com/datasets/msambare/fer2013). 
Download it from [Google Drive] (link below) and place folders into respective folders in repostiory. 

Google Drive Dataset Link (https://drive.google.com/drive/folders/1YF8jsAFd8G6g-6gJU4M9S_6A5ih-Djcz?usp=drive_link).

STEPS TO LOAD DATA

- Download each folder test/train/validation from google drive
- Insert each folder into their respective folder in repository
- The data will be saved locally on your machine and no data(test/train/validation) should be pushed to repository

## More on Data Prep
- [DATA PREPARATION GUIDE](DATA_PREP_GUIDE.MD)


## Running the model

Steps to train the model

    -From the project root directory, run: python .\Scripts\train_eval.py --dir ".\data\processed" --epochs 27 --batch_size 64 --save_model
    - --dir specifies the folder containing the train/, val/, and test/ splits.
    - --epochs controls how many training epochs are performed.
    - --batch_size sets the mini-batch size.
    - --save_model tells the script to save the trained weights (e.g., emotion_cnn.pt) into the results/ directory.

## Model 

Our CNN takes a set of 48x48 grayscale images and runs them through 4 convolutional layers with a kernel size of 3, padding of 1, and stride of 1.
These layers use ReLU activation as well as max pooling. After the final convolutional layer a dropout function is used to prevent overfitting.
We used the adamW optimizer and cross-entropy loss function for their ease of use and mix of efficacy and performance.

Our emotion recognition model is a **Convolutional Neural Network (CNN)** designed for 48×48 grayscale facial images.

## Analysis
How to Run Analysis

    -python .\Scripts\data_analysis.py --dir ".\data\processed" --out ".\results\analysis"

    -The analysis script will automatically locate the latest model weights, evaluate the CNN on the test set, and generate:

        - Confusion matrices (counts and normalized)
        - Per-class accuracy plots
        - Top-k accuracy metrics
        - Classification reports (precision, recall, F1)
        - Dataset distribution figures and summary CSV files

    All outputs are written to the results/analysis directory for use in reports and presentations.

## AddAnythingElse
