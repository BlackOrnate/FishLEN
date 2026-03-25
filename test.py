import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from FLN import FishLengthNet
from tools import create_dataloaders

from collections import defaultdict
import argparse
from tools import save_mask, save_len_pred, draw_test_result


def infer_loop(models, dataloader, save_path, title_name, image_path):
    for name, model in models.items():
        model.eval()

    fish_info_list = []

    with torch.no_grad():
        for images, masks, lengths, features, fish_infos in tqdm(dataloader, desc="Inference"):
            model = models["mask_pred"]
            images = images.to(device)

            mask_pred_outputs = model(images)

            model = models["len_pred"]
            features = features.to(device)
            lengths = lengths.to(device)

            len_pred_outputs = model(mask_pred_outputs, features)

            save_mask(image_path, fish_infos["fish_name"], images, mask_pred_outputs, len_pred_outputs, lengths, save_path)

            fish_infos["len_pred_values"] = len_pred_outputs.cpu().squeeze().tolist()
            fish_infos["len_true_values"] = lengths.cpu().squeeze().tolist()
            fish_info_list.append(fish_infos)

        merged = defaultdict(list)
        for d in fish_info_list:
            for k, v in d.items():
                merged[k].extend(v)
        fish_info_dict = dict(merged)

        save_len_pred(fish_info_dict, save_path)
        # draw_test_result(fish_info_dict, title_name)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="./Dataset/AC I/full/images_padded_output")
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--csv_path", type=str, default="./Dataset/AC I/aging_cohort_I.csv")
    parser.add_argument("--save_path", type=str, default="./results")

    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--weights_pths", nargs='+', type=str, default=[])
    parser.add_argument("--title_name", type=str, default="Grid + Black Background")
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
    weights_pths = args.weights_pths
    title_name = args.title_name
    day_indexes = args.day_indexes
    extra_info_column_names = args.extra_info_column_names

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    models = {
        "mask_pred": FishLengthNet(model_type="mask_pred"),
        "len_pred": FishLengthNet(model_type="len_pred" if len(extra_info_column_names) == 0 else "len_pred_new",
                                  num_classes=len(extra_info_column_names))
    }

    for name, model in models.items():
        models[name] = model.to(device)

    if len(weights_pths) == 0:
        weights_pths.append(f"{save_path}/mask_pred.pth")
        weights_pths.append(f"{save_path}/len_pred.pth")

    for model_type, weighs_pth in zip(models, weights_pths):
        models[model_type].load_state_dict(torch.load(weighs_pth, weights_only=True))

    dataloader, *_ = create_dataloaders(image_path=image_path,
                                        mask_path=mask_path,
                                        csv_path=csv_path,
                                        seq_length=5,
                                        batch_size=batch_size,
                                        day_indexes=day_indexes,
                                        used="Test",
                                        extra_info_column_names=extra_info_column_names)

    infer_loop(models, dataloader, save_path, title_name, image_path)
