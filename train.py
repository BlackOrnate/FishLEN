import os
import numpy as np
import torch
from torch import optim
from tools import create_dataloaders, check_result
from FLN import FishLengthNet
from loss import loss_fn
from tqdm import tqdm
import shutil
import argparse


def train_loop(epoch_size, models, save_path, train_dataloader, valid_dataloader, optimizers=None, loss_fn=None):
    min_mask_pred_loss = 1000
    min_len_pred_loss = 1000
    min_len_pred_less_half_num = 0
    min_len_pred_less_one_num = 0
    best_pth_name_dict = {
        "mask_pred": "",
        "len_pred": "",
    }

    for epoch in range(1, epoch_size + 1):
        print(f"\nEpoch {epoch}/{epoch_size}")

        for name, model in models.items():
            model.train()

        train_mask_pred_loss = 0.0
        train_len_pred_loss = 0.0
        total_samples = 0

        for images, masks, lengths, features, fish_infos in tqdm(train_dataloader, desc="Training"):
            model = models["mask_pred"]
            images = images.to(device)
            masks = masks.to(device)

            mask_pred_outputs = model(images)
            mask_pred_loss = loss_fn(model.model_type, mask_pred_outputs, masks)

            model = models["len_pred"]
            features = features.to(device)
            lengths = lengths.to(device)

            len_pred_outputs = model(mask_pred_outputs.detach(), features)
            len_pred_loss = loss_fn(model.model_type, len_pred_outputs, lengths)

            optimizers["mask_pred"].zero_grad()
            mask_pred_loss.backward()
            optimizers["mask_pred"].step()

            optimizers["len_pred"].zero_grad()
            len_pred_loss.backward()
            optimizers["len_pred"].step()

            current_batch_size = images.size(0)
            train_mask_pred_loss += mask_pred_loss.item() * current_batch_size
            train_len_pred_loss += len_pred_loss.item() * current_batch_size
            total_samples += current_batch_size

        avg_train_mask_pred_loss = train_mask_pred_loss / total_samples
        avg_train_len_pred_loss = train_len_pred_loss / total_samples
        print(
            f"Epoch {epoch}/{epoch_size} - Training Mask Prediction Loss: {avg_train_mask_pred_loss:.4f} - Training Length Prediction Loss: {avg_train_len_pred_loss:.4f}")

        for name, model in models.items():
            model.eval()
        valid_mask_pred_loss = 0.0
        valid_len_pred_loss = 0.0
        total_samples = 0

        len_pred_values = []
        len_true_values = []
        name_list = []
        image_list = []
        original_mask_list = []
        pred_mask_list = []

        with torch.no_grad():
            for images, masks, lengths, features, fish_infos in tqdm(valid_dataloader, desc="Validation"):
                model = models["mask_pred"]
                images = images.to(device)
                masks = masks.to(device)

                mask_pred_outputs = model(images)
                mask_pred_loss = loss_fn(model.model_type, mask_pred_outputs, masks)

                model = models["len_pred"]
                features = features.to(device)
                lengths = lengths.to(device)

                len_pred_outputs = model(mask_pred_outputs.detach(), features)
                len_pred_loss = loss_fn(model.model_type, len_pred_outputs, lengths)

                current_batch_size = images.size(0)
                valid_mask_pred_loss += mask_pred_loss.item() * current_batch_size
                valid_len_pred_loss += len_pred_loss.item() * current_batch_size
                total_samples += current_batch_size

                len_pred_values.extend(len_pred_outputs.cpu().numpy())
                len_true_values.extend(lengths.cpu().numpy())
                name_list.extend(fish_infos['fish_name'])
                image_list.extend(images.cpu())
                original_mask_list.extend(masks.cpu())
                pred_mask_list.extend(mask_pred_outputs.cpu())

        avg_valid_mask_pred_loss = valid_mask_pred_loss / total_samples
        avg_valid_len_pred_loss = valid_len_pred_loss / total_samples
        print(
            f"Epoch {epoch}/{epoch_size} - Validation Mask Prediction Loss: {avg_valid_mask_pred_loss:.4f} - Validation Length Prediction Loss: {avg_valid_len_pred_loss:.4f}")

        len_pred_values = np.array(len_pred_values)
        len_true_values = np.array(len_true_values)
        current_len_pred_less_one_num = (np.abs(len_pred_values - len_true_values) / len_true_values < 0.1).sum()
        current_len_pred_less_half_num = (np.abs(len_pred_values - len_true_values) / len_true_values < 0.05).sum()

        if min_mask_pred_loss > avg_valid_mask_pred_loss or min_len_pred_less_one_num < current_len_pred_less_one_num or min_len_pred_less_half_num < current_len_pred_less_half_num:
            check_list = {"mask_pred": min_mask_pred_loss > avg_valid_mask_pred_loss,
                          "len_pred_less_one": min_len_pred_less_one_num < current_len_pred_less_one_num,
                          "len_pred_less_half": min_len_pred_less_half_num < current_len_pred_less_half_num}
            best_pth_name_dict = check_result(models, image_list, original_mask_list, pred_mask_list, name_list, epoch,
                                              save_plt, save_pth, len_pred_values, len_true_values,
                                              current_len_pred_less_one_num, current_len_pred_less_half_num, check_list,
                                              best_pth_name_dict, save_path)
            min_mask_pred_loss = avg_valid_mask_pred_loss if min_mask_pred_loss > avg_valid_mask_pred_loss else min_mask_pred_loss
            min_len_pred_loss = avg_valid_len_pred_loss if min_len_pred_loss > avg_valid_len_pred_loss else min_len_pred_loss
            min_len_pred_less_one_num = current_len_pred_less_one_num if min_len_pred_less_one_num < current_len_pred_less_one_num else min_len_pred_less_one_num
            min_len_pred_less_half_num = current_len_pred_less_half_num if min_len_pred_less_half_num < current_len_pred_less_half_num else min_len_pred_less_half_num

    for key, value in best_pth_name_dict.items():
        dst = f"{save_path}/{key}.pth"
        shutil.copy(value, dst)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="./Dataset/AC I/train/Images_padded_output")
    parser.add_argument("--mask_path", type=str, default="./Dataset/AC I/train/Masks_padded_output")
    parser.add_argument("--csv_path", type=str, default="./Dataset/AC I/aging_cohort_I.csv")
    parser.add_argument("--save_path", type=str, default="./result")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epoch_size", type=int, default=1000)
    parser.add_argument("--stop_train_epoch_num", type=int, default=150)
    parser.add_argument("--save_plt", type=bool, default=True)
    parser.add_argument("--save_pth", type=bool, default=True)
    parser.add_argument("--weights_pths", nargs='+', type=str, default=None)
    parser.add_argument("--day_indexes", nargs='+', type=int, default=None)
    parser.add_argument("--extra_info_column_names", nargs='+', type=str, default=[])
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    image_path = args.image_path
    mask_path = args.mask_path
    csv_path = args.csv_path
    save_path = args.save_path

    batch_size = args.batch_size
    epoch_size = args.epoch_size
    stop_train_epoch_num = args.stop_train_epoch_num

    save_plt = args.save_plt
    save_pth = args.save_pth

    weights_pths = args.weights_pths
    day_indexes = args.day_indexes
    extra_info_column_names = args.extra_info_column_names

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataloader, train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(image_path=image_path,
                                                                                         mask_path=mask_path,
                                                                                         csv_path=csv_path,
                                                                                         seq_length=5,
                                                                                         batch_size=batch_size,
                                                                                         day_indexes=day_indexes,
                                                                                         used="Train",
                                                                                         extra_info_column_names=extra_info_column_names)

    models = {
        "mask_pred": FishLengthNet(model_type="mask_pred"),
        "len_pred": FishLengthNet(model_type="len_pred" if len(extra_info_column_names) == 0 else "len_pred_new",
                                  num_classes=len(extra_info_column_names)),
    }

    optimizers = {}
    if weights_pths is not None:
        for model_type, weighs_pth in zip(models, weights_pths):
            models[model_type] = models[model_type].to(device)
            if weighs_pth != "":
                models[model_type].load_state_dict(torch.load(weighs_pth, weights_only=True))
            optimizers[model_type] = optim.Adam(models[model_type].parameters(), lr=1e-4, betas=(0.9, 0.999),
                                                weight_decay=1e-4)
    else:
        for model_type in models:
            models[model_type] = models[model_type].to(device)
            optimizers[model_type] = optim.Adam(models[model_type].parameters(), lr=1e-4, betas=(0.9, 0.999),
                                                weight_decay=1e-4)

    train_loop(epoch_size, models, save_path, train_dataloader, valid_dataloader, optimizers, loss_fn)
