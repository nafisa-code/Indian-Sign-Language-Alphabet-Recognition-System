import os
import shutil
import random
BASE_DIR = r"C:\ISL\dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
print("Starting dataset split...")
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)
for label in os.listdir(RAW_DIR):
    label_path = os.path.join(RAW_DIR, label)
    if not os.path.isdir(label_path):
        continue
    images = os.listdir(label_path)
    random.shuffle(images)
    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    splits = {
        TRAIN_DIR: images[:train_end],
        VAL_DIR: images[train_end:val_end],
        TEST_DIR: images[val_end:]
    }
    for split_dir, split_images in splits.items():
        split_label_dir = os.path.join(split_dir, label)
        os.makedirs(split_label_dir, exist_ok=True)
        for img in split_images:
            shutil.copy(
                os.path.join(label_path, img),
                os.path.join(split_label_dir, img)
            )
    print(f"Done: {label}")
print("DATASET SPLIT COMPLETED")
input("Press Enter to exit...")
