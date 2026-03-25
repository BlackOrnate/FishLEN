import os
import random
import cv2
import numpy as np
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

data_root = "../data"  # Path for the source images
output_root = "./SAM/masks_output"  # Path for the SAM's result

sam_checkpoint = "sam_vit_l_0b3195.pth"  # SAM checkpoint
model_type = "vit_l"  # SAM model type

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading SAM model...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)
print("Model loaded successfully.")


def get_random_subset(file_list, min_ratio=0.1, max_ratio=0.2):
    count = len(file_list)
    if count == 0:
        return []
    num_to_select = random.randint(max(1, int(count * min_ratio)), max(1, int(count * max_ratio)))
    return random.sample(file_list, num_to_select)


def process_image(image_path, output_folder):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: failed to read {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image_rgb)

    os.makedirs(output_folder, exist_ok=True)
    for idx, m in enumerate(masks):
        mask = (m['segmentation'].astype(np.uint8) * 255)
        save_path = os.path.join(output_folder, f"mask_{idx:03d}.png")
        cv2.imwrite(save_path, mask)


for root, dirs, files in os.walk(data_root):
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        continue

    selected_images = get_random_subset(image_files)
    print(f"Processing {len(selected_images)} images in {root}...")

    for image_name in tqdm(selected_images, desc=f"Processing {root}"):
        image_path = os.path.join(root, image_name)

        rel_path = os.path.relpath(root, data_root)
        image_stem = os.path.splitext(image_name)[0]
        output_folder = os.path.join(output_root, rel_path, image_stem)

        process_image(image_path, output_folder)

print("All done.")
