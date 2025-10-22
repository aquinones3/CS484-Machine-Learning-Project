import os 
from pathlib import Path
import shutil
import random
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parents[1]

base_directory = ROOT / "data" / "raw" 
train_directory = base_directory / "train"
validation_directory = base_directory / "val"
os.makedirs(validation_directory, exist_ok=True) # creating validation directory


validation_split_ration = 0.15 # 15% of train data for validation
random.seed(42)


for emotion in os.listdir(train_directory): #iteratre through all emtion folders(angry, happy etc)
    emotion_path = os.path.join(train_directory /emotion)
    if not emotion_path.is_dir(): # skipping if not a folder
        continue


    images = os.listdir(emotion_path)
    train_images, validation_images = train_test_split(images,test_size=validation_split_ration, random_state=42)
    
    
    validation_emotion_path =validation_directory / emotion
    os.makedir(validation_emotion_path, exist_ok=True)

    for img in validation_images:
        src = emotion_path / img
        dst = validation_emotion_path / img
        shutil.copy(src, dst)

    print(f"{emotion}: moved {len(validation_images)} images to validation folder.")
