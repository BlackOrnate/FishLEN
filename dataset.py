import os
import numpy as np
import torch
from numpy import dtype
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re


def check_image_mask_exists(images_path, masks_path, fish_info, used):
    valid_rows = []
    all_folders = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]

    if used == "Train":
        for _, row in fish_info.iterrows():
            day_index = str(row["day_index"])

            final_image = row["final_image"] + ".png"

            matched_folder = [folder for folder in all_folders if folder.startswith(f"Day {day_index} Week")]

            if not matched_folder:
                continue
            else:
                matched_folder = matched_folder[0]

            image_folder_path = os.path.join(images_path, matched_folder)
            mask_folder_path = os.path.join(masks_path, matched_folder)

            if final_image in os.listdir(image_folder_path) and final_image in os.listdir(mask_folder_path):
                new_row = row.copy()
                new_row["day_index"] = day_index
                new_row["fish_id"] = new_row['fish_no']
                new_row["fish_name"] = final_image
                new_row["image_path"] = os.path.join(image_folder_path, final_image)
                new_row["mask_path"] = os.path.join(mask_folder_path, final_image)

                valid_rows.append(new_row)
        fish_info = pd.DataFrame(valid_rows)
    else:
        if fish_info.empty:
            for day_index in all_folders:
                if any(entry.is_dir() for entry in os.scandir(os.path.join(images_path, day_index))):
                    for fish_id in os.listdir(os.path.join(images_path, day_index)):
                        for image_name in os.listdir(os.path.join(images_path, day_index, fish_id)):
                            row = pd.DataFrame([{
                                "day_index": day_index,
                                "fish_id": fish_id,
                                "fish_name": image_name,
                                "image_path": os.path.join(images_path, day_index, fish_id, image_name),
                                "mask_path": os.path.join(masks_path, day_index, fish_id,
                                                          image_name) if masks_path != "" else "",
                                "label": "model_pred",
                            }])
                            valid_rows.append(row)
                else:
                    for image_name in os.listdir(os.path.join(images_path, day_index)):
                        row = pd.DataFrame([{
                            "day_index": day_index,
                            "fish_name": image_name,
                            "image_path": os.path.join(images_path, day_index, image_name),
                            "mask_path": os.path.join(masks_path, day_index, image_name) if masks_path != "" else "",
                            "label": "model_pred",
                        }])
                        valid_rows.append(row)
            fish_info = pd.concat(valid_rows, ignore_index=True)
        else:
            for _, row in fish_info.iterrows():
                day_index = str(row["day_index"])
                matched = None
                for f in os.listdir(images_path):
                    match = re.search(r"Day\s+(\d+)", f)
                    if match:
                        day_num = match.group(1)
                        if day_num == day_index:
                            matched = f
                            break

                if matched is None:
                    continue

                day_index = matched

                if not os.path.isdir(os.path.join(images_path, day_index)):
                    continue

                if any(entry.is_dir() for entry in os.scandir(os.path.join(images_path, day_index))):
                    for fish_id in os.listdir(os.path.join(images_path, day_index)):
                        if int(fish_id) != row["fish_no"]:
                            continue

                        for image_name in os.listdir(os.path.join(images_path, day_index, fish_id)):
                            new_row = row.copy()
                            new_row["day_index"] = day_num
                            new_row["fish_id"] = fish_id
                            new_row["fish_name"] = image_name
                            new_row["image_path"] = os.path.join(images_path, day_index, fish_id, image_name)
                            new_row["mask_path"] = os.path.join(masks_path, day_index, fish_id,
                                                                image_name) if masks_path != "" else ""
                            # new_row["label"] = "model_pred_select" if row["image_name"] == image_name else "model_pred"
                            new_row["label"] = "model_pred_select" if row[
                                                                          "final_image"] + '.png' == image_name else "model_pred"
                            valid_rows.append(new_row)
                else:
                    for image_name in os.listdir(os.path.join(images_path, day_index)):
                        if row["final_image"] + '.png' == image_name:
                            new_row = row.copy()
                            new_row["day_index"] = day_num
                            new_row["fish_id"] = str(new_row["fish_no"])
                            new_row["fish_name"] = image_name
                            new_row["image_path"] = os.path.join(images_path, day_index, image_name)
                            new_row["mask_path"] = os.path.join(masks_path, day_index,
                                                                image_name) if masks_path != "" else ""
                            # new_row["label"] = "model_pred_select" if row["image_name"] == image_name else "model_pred"
                            new_row["label"] = "model_pred_select" if row[
                                                                          "final_image"] + '.png' == image_name else "model_pred"
                            valid_rows.append(new_row)
                            break
            fish_info = pd.DataFrame(valid_rows)
    return fish_info


def check_col_exist(extra_info_colum_names, fish_info):
    missing_cols = []
    for col in extra_info_colum_names:
        if col not in fish_info.columns:
            missing_cols.append(col)

    if missing_cols:
        raise ValueError(f"These columns are missing in csv: {missing_cols}")


class FishDataset(Dataset):
    def __init__(self, images_path: str, mask_path: str, csv_path: str, seq_length: int, day_indexes=None,
                 transform=None, used="Train", extra_info_column_names=None):
        self.images_path = images_path
        self.mask_path = mask_path
        self.csv_path = csv_path
        self.seq_length = seq_length
        self.transform = transform if used == "Train" else None
        self.used = used
        self.extra_info_column_names = extra_info_column_names

        if csv_path != "":
            fish_info = pd.read_csv(csv_path)
            if day_indexes is not None:
                fish_info = fish_info[fish_info["day_index"].isin(day_indexes)]
        else:
            fish_info = pd.DataFrame()

        fish_info = check_image_mask_exists(images_path, mask_path, fish_info, used)

        self.fish_info = fish_info

        check_col_exist(self.extra_info_column_names, self.fish_info)

        self.category_maps = {}
        for col in self.extra_info_column_names:
            if fish_info[col].dtype == object:
                unique_values = sorted(fish_info[col].dropna().unique().tolist())
                self.category_maps[col] = {v: i for i, v in enumerate(unique_values)}

    def __len__(self):
        return len(self.fish_info)

    def __getitem__(self, index):
        if len(self.extra_info_column_names) == 0:
            fish_info = self.fish_info.iloc[index]
            image_path = fish_info["image_path"]
            mask_path = fish_info["mask_path"]

            image = np.array(Image.open(image_path).convert("RGB"))

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask).astype(np.float32)
                mask[mask == 255.0] = 1.0
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if self.transform:
                augmented = self.transform["geometry"](image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                image = self.transform["color"](image=image)["image"]

            image = torch.tensor(np.clip(image, 0, 255).astype(np.uint8).transpose(2, 0, 1))
            mask = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)

            length = torch.tensor(fish_info["avg_length_mm"] if "avg_length_mm" in fish_info else -1,
                                  dtype=torch.float).unsqueeze(0)
            features = torch.tensor(0).unsqueeze(0)

            return image, mask, length, features, fish_info.to_dict()
        else:
            fish_info = self.fish_info.iloc[index]
            image_path = fish_info["image_path"]
            mask_path = fish_info["mask_path"]

            image = np.array(Image.open(image_path).convert("RGB"))

            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
                mask = np.array(mask).astype(np.float32)
                mask[mask == 255.0] = 1.0
            else:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            if self.transform:
                augmented = self.transform["geometry"](image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                image = self.transform["color"](image=image)["image"]

            image = torch.tensor(np.clip(image, 0, 255).astype(np.uint8).transpose(2, 0, 1))
            mask = torch.tensor(mask, dtype=torch.int64).unsqueeze(0)

            length = torch.tensor(fish_info["avg_length_mm"] if "avg_length_mm" in fish_info else -1,
                                  dtype=torch.float).unsqueeze(0)

            extra_info_values = []
            for col in self.extra_info_column_names:
                value = fish_info[col]

                if col in self.category_maps:
                    value = self.category_maps[col][value]

                extra_info_values.append(float(value))

            features = torch.tensor(extra_info_values, dtype=torch.float32)

            return image, mask, length, features, fish_info.to_dict()
