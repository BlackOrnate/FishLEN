import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from dataset import FishDataset
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

torch.set_printoptions(precision=16)


def create_dataloaders(image_path: str, mask_path: str, csv_path: str, seq_length: int, batch_size: int, day_indexes,
                       used: str = "Train", extra_info_column_names: list = None):
    preprocess = {
        "geometry": A.Compose([
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                shear=(-5, 5),
                rotate=(-179, 179),
                interpolation=0,
                border_mode=0,
                p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ], additional_targets={'mask': 'mask'}),

        "color": A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=(1, 3), p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
                A.GaussNoise(std_range=(np.sqrt(10 / 255), np.sqrt(30 / 255)), mean_range=(0.0, 0.0), p=0.5),
            ], p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.25,
                saturation=0.2,
                hue=0.03,
                p=0.5)
        ])
    }

    full_dataset = FishDataset(image_path, mask_path, csv_path, seq_length, day_indexes, preprocess, used,
                               extra_info_column_names)

    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.2 * total_len)
    test_len = total_len - train_len - val_len

    train_data, val_data, test_data = random_split(full_dataset, [train_len, val_len, test_len])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                 persistent_workers=True)

    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)

    return dataloader, train_dataloader, valid_dataloader, test_dataloader


def check_result(models, images, original_masks, pred_masks, names, epoch, save_plt, save_pth, all_predictions,
                 all_values, len_less_one_num, len_less_half_num, check_list, best_pth_name_dict, save_path):
    already_plt = False
    if check_list is not None:
        if check_list["mask_pred"]:
            plt_mask(names, images, original_masks, pred_masks, epoch, save_plt, save_path)
            if save_pth:
                os.makedirs(f"{save_path}/pth_output/mask_pred/", exist_ok=True)
                torch.save(models["mask_pred"].state_dict(), f"{save_path}/pth_output/mask_pred/Epoch_{epoch}.pth")

            best_pth_name_dict['mask_pred'] = f"{save_path}/pth_output/mask_pred/Epoch_{epoch}.pth"

        if check_list["len_pred_less_one"]:
            draw_result(all_predictions, all_values, len_less_one_num, len_less_half_num, epoch, save_plt, save_path)
            already_plt = True
            if save_pth:
                os.makedirs(f"{save_path}/pth_output/len_pred/", exist_ok=True)
                torch.save(models["len_pred"].state_dict(),
                           f"{save_path}/pth_output/len_pred/Epoch_{epoch}_len_pred_less_one_{len_less_one_num}.pth")

            best_pth_name_dict[
                'len_pred'] = f"{save_path}/pth_output/len_pred/Epoch_{epoch}_len_pred_less_one_{len_less_one_num}.pth"

        if check_list["len_pred_less_half"]:
            if not already_plt:
                draw_result(all_predictions, all_values, len_less_one_num, len_less_half_num, epoch, save_plt,
                            save_path)
            if save_pth:
                os.makedirs(f"{save_path}/pth_output/len_pred/", exist_ok=True)
                torch.save(models["len_pred"].state_dict(),
                           f"{save_path}/pth_output/len_pred/Epoch_{epoch}_len_less_half_num_{len_less_half_num}.pth")

            best_pth_name_dict[
                'len_pred'] = f"{save_path}/pth_output/len_pred/Epoch_{epoch}_len_less_half_num_{len_less_half_num}.pth"

    return best_pth_name_dict


def plt_mask(names, images, original_masks, pred_masks, epoch, save_plt, save_path):
    idx = 0
    sample_images = images[idx].cpu().numpy()
    sample_masks = original_masks[idx].cpu().numpy()
    predicted_masks = pred_masks[idx].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(sample_images.transpose(1, 2, 0))
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(sample_masks.squeeze(), cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')

    ax[2].imshow(predicted_masks.squeeze(), cmap='gray')
    ax[2].set_title('Predicted Mask')
    ax[2].axis('off')

    plt.suptitle(f'{names[idx]} Epoch {epoch}')
    if save_plt:
        os.makedirs(f"{save_path}/plt_output/mask_pred/", exist_ok=True)
        plt.savefig(f"{save_path}/plt_output/mask_pred/Epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()


def draw_result(all_predictions, all_values, len_less_one_num, len_less_half_num, epoch, save_plt, save_path):
    all_predictions = np.array(all_predictions)
    pred_length = all_predictions.flatten()

    all_values = np.array(all_values)
    gt_length = all_values.flatten()

    image_ids = np.arange(len(all_values))

    plt.figure(figsize=(10, 5))

    plt.scatter(image_ids, gt_length, color='blue', label="True Values", marker='o', alpha=0.6)
    plt.scatter(image_ids, pred_length, color='red', label="Predicted Values", marker='^', alpha=0.6)

    plt.xlabel("Image ID")
    plt.ylabel("Fish Length")
    plt.title(
        f"True vs Predicted Fish Length: Epoch {epoch} \n Less than 10% length: {len_less_one_num / len(all_predictions) * 100:.2f}%     Less than 5% length: {len_less_half_num / len(all_predictions) * 100:.2f}%")
    plt.legend()
    plt.grid(True)

    if save_plt:
        os.makedirs(f"{save_path}/plt_output/len_pred/", exist_ok=True)
        plt.savefig(f"{save_path}/plt_output/len_pred/Epoch_{epoch}.png")
        plt.close()
    else:
        plt.show()


def draw_error_plot(all_predictions, all_values):
    all_predictions = np.array(all_predictions).flatten()
    all_values = np.array(all_values).flatten()

    errors = all_values - all_predictions

    image_ids = np.arange(len(all_values))

    colors = ['blue' if err > 0 else 'red' for err in errors]

    plt.figure(figsize=(10, 5))
    plt.plot(image_ids, errors, marker='o', linestyle='-', color=colors, label="Error")
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.xlabel("Image ID")
    plt.ylabel("Error (True - Predicted)")
    plt.title("Prediction Error per Image")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()



def save_mask(image_path, names, images, outputs, len_pred_values, len_true_values, save_path):
    sample_images = images.cpu().numpy()
    predicted_masks = outputs.cpu().numpy()
    len_pred_values = len_pred_values.cpu().numpy()
    len_true_values = len_true_values.cpu().numpy()

    folder_names = []
    for folder_name in os.listdir(image_path):
        file_names = os.listdir(os.path.join(image_path, folder_name))
        for name in names:
            if name in file_names:
                folder_names.append(folder_name)

    base_output_dir = f"{save_path}/predicted_mask_test"

    for idx in range(len(sample_images)):
        input_img = sample_images[idx].transpose(1, 2, 0)
        mask = predicted_masks[idx].squeeze()
        image_name = names[idx].split(".png")[0]
        folder_name = folder_names[idx]

        folder_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(folder_dir, exist_ok=True)
        plt.imsave(f"{folder_dir}/{image_name}_input.png", input_img)
        plt.imsave(f"{folder_dir}/{image_name}_mask.png", mask, cmap='gray')

        print(f"Saved: {names[idx]}_input.png and {names[idx]}_mask.png")

        pred_len = len_pred_values[idx][0]
        true_len = len_true_values[idx][0]
        if true_len == 0:
            continue
        relative_error = float(abs(pred_len - true_len) / true_len)

        if relative_error > 0.10:
            error_level = "above_10_percent"
        elif relative_error > 0.05:
            error_level = "between_5_and_10_percent"
        else:
            error_level = "below_5_percent"

        error_save_dir = os.path.join(f"{save_path}/predicted_mask_error", error_level, folder_name)
        os.makedirs(error_save_dir, exist_ok=True)

        plt.imsave(f"{error_save_dir}/{image_name}_input.png", input_img)
        plt.imsave(f"{error_save_dir}/{image_name}_mask.png", mask, cmap='gray')

        print(f"Saved: {image_name} with error {relative_error:.2%} into {error_save_dir}")


def save_len_pred(fish_info_dict, save_path):
    result_data = {
        'day_index': [f for f in fish_info_dict["day_index"]],
        'image_name': [f.split('.png')[0] for f in fish_info_dict["fish_name"]],
        'gt_length': [f for f in fish_info_dict["len_true_values"]],
        'pred_length': [f for f in fish_info_dict["len_pred_values"]],
        'type': [f for f in fish_info_dict["label"]],
        'image_path': [f for f in fish_info_dict["image_path"]],
    }

    if "fish_id" in fish_info_dict:
        result_data["fish_id"] = fish_info_dict["fish_id"]

    df = pd.DataFrame(result_data)

    df.to_csv(f"{save_path}/result.csv", index=False)


def draw_test_result(fish_info_dict, title_name):
    # Iterate over each unique day index while preserving the original order
    for dat_index in list(dict.fromkeys(fish_info_dict['day_index'])):
        df = pd.DataFrame(fish_info_dict)
        df = df[df["day_index"] == dat_index][['fish_id', 'label', 'len_pred_values', 'len_true_values']]

        # Predictions from the same model using all available images per fish
        all_images_pred_df = df[df["label"] == "model_pred"].copy()

        # Predictions from the same model using the best selected image per fish
        best_image_pred_df = df[df["label"] == "model_pred_select"].copy()

        # Ground-truth mean length per fish ID
        true_per_id = df.groupby("fish_id", sort=True)["len_true_values"].mean()

        # Mean prediction per fish ID across all images
        mean_pred_all_images = all_images_pred_df.groupby("fish_id", sort=True)["len_pred_values"].mean()

        # One prediction per fish ID from the best selected image
        pred_best_image_per_id = best_image_pred_df.drop_duplicates("fish_id").set_index("fish_id")["len_pred_values"]

        all_ids = set(df["fish_id"])

        def sort_key(fid):
            try:
                return (0, float(fid))
            except Exception:
                return (1, str(fid))

        unique_ids_sorted = sorted(all_ids, key=sort_key)
        x_map = {fid: i for i, fid in enumerate(unique_ids_sorted)}
        x_positions = [x_map[fid] for fid in unique_ids_sorted]

        plt.figure(figsize=(12, 6))

        # All image-level predictions (red triangles)
        if not all_images_pred_df.empty:
            plt.scatter(
                all_images_pred_df["fish_id"].map(x_map),
                all_images_pred_df["len_pred_values"],
                marker="^", s=40, color="red", alpha=0.6,
                label="Model Predictions (from all images)"
            )

        # Mean prediction across all images (yellow circles)
        y_pred_mean = mean_pred_all_images.reindex(unique_ids_sorted).values
        plt.scatter(
            x_positions, y_pred_mean,
            marker='o', s=70, color='yellow', edgecolors='k', linewidths=0.5,
            label='Model Prediction (mean)'
        )

        # Ground-truth values (manual measurements)
        y_true = true_per_id.reindex(unique_ids_sorted).values
        plt.scatter(
            x_positions, y_true,
            marker='o', s=70, color='blue', edgecolors='k', linewidths=0.5,
            label='Manual Measurement (from the best selected image)'
        )

        # Prediction from the best selected image (green circles)
        y_other = pred_best_image_per_id.reindex(unique_ids_sorted).values
        plt.scatter(
            x_positions, y_other,
            marker='o', s=70, color='green', edgecolors='k', linewidths=0.5,
            label='Model Prediction (from best selected image)'
        )

        num_images_per_fish = all_images_pred_df.groupby('fish_id').size()
        labels = [f"{fid}\n(n={num_images_per_fish.get(fid, 0)})" for fid in unique_ids_sorted]

        # Compute statistics to visualize image-level prediction variability
        # (IQR and min–max range) from the same model
        pred_stats_all_images = all_images_pred_df.groupby('fish_id')['len_pred_values'].agg(
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            minv='min',
            maxv='max'
        ).reindex(unique_ids_sorted)

        q1 = pred_stats_all_images['q1'].values
        q3 = pred_stats_all_images['q3'].values
        minv = pred_stats_all_images['minv'].values
        maxv = pred_stats_all_images['maxv'].values

        iqr_height = q3 - q1
        upper_err = maxv - q3
        lower_err = q3 - minv
        yerr = np.vstack([lower_err, upper_err])

        plt.bar(
            x_positions,
            iqr_height,
            bottom=q1,
            yerr=yerr,
            color='#ff9999',
            edgecolor='red',
            alpha=0.35,
            capsize=6,
            linewidth=1.0,
            zorder=1
        )

        plt.xticks(ticks=x_positions, labels=labels, rotation=0)
        plt.xlabel("Fish ID")
        plt.ylabel("Fish Length (mm)")
        plt.title(f"{title_name} - {dat_index}")
        # plt.grid(True, linestyle='--', alpha=0.3)
        plt.ylim(25, 100)
        # plt.ylim(25, 140)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
