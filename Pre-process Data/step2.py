import os
import cv2
import numpy as np
from tqdm import tqdm

# 源目录和目标目录
# source_root = "./masks_output"
# target_root = "./masks_padded_output"
# source_root = "/data/hongrui/FishLengthNet/Test dataset/aging cohort 1 test samples v2"
# target_root = "/data/hongrui/FishLengthNet/Test dataset/data_padded_output v2"
source_root = "/data/hongrui/FishLengthNet/Test dataset/AC_4"
target_root = "/data/hongrui/FishLengthNet/Test dataset/data_padded_output AC_4"

os.makedirs(target_root, exist_ok=True)

for root, dirs, files in os.walk(source_root):
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        continue

    rel_path = os.path.relpath(root, source_root)
    target_dir = os.path.join(target_root, rel_path)
    os.makedirs(target_dir, exist_ok=True)

    for image_file in tqdm(image_files, desc=f"Processing {rel_path}"):
        source_image_path = os.path.join(root, image_file)
        target_image_path = os.path.join(target_dir, image_file)

        image = cv2.imread(source_image_path)
        if image is None:
            print(f"Warning: failed to read {source_image_path}")
            continue

        h, w, c = image.shape
        if h == 1088 and w == 1920:
            cv2.imwrite(target_image_path, image)
            continue

        padded_image = np.zeros((1088, 1920, c), dtype=image.dtype)
        padded_image[:h, :w, :] = image

        cv2.imwrite(target_image_path, padded_image)

print("All images have been processed and saved to", target_root)
