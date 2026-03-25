import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet34_Weights


class FishLengthNet(nn.Module):
    def __init__(self, model_type, num_classes=0):
        super(FishLengthNet, self).__init__()
        self.model_type = model_type

        if model_type == "mask_pred":
            backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            self.encoder_layers = list(backbone.children())[:-2]

            self.conv1 = nn.Sequential(*self.encoder_layers[:3])  # Conv1
            self.enc1 = nn.Sequential(*self.encoder_layers[3:5])  # Conv2_x
            self.enc2 = self.encoder_layers[5]  # Conv3_x
            self.enc3 = self.encoder_layers[6]  # Conv4_x
            self.enc4 = self.encoder_layers[7]  # Conv5_x

            # Bottleneck
            self.bottleneck = nn.Conv2d(512, 512, kernel_size=3, padding=1)

            # Decoder with Skip Connections and Conv1x1 for channel adjustment
            self.dec4 = nn.Sequential(
                UpSample2x(),
                nn.Conv2d(512, 256, kernel_size=1)
            )

            self.dec3 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                UpSample2x(),
                nn.Conv2d(256, 128, kernel_size=1)
            )

            self.dec2 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                UpSample2x(),
                nn.Conv2d(128, 64, kernel_size=1)
            )

            self.dec1 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1),
                UpSample2x(),
                nn.Conv2d(64, 64, kernel_size=1)
            )

            # 最后一层
            self.final_conv = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1),
                UpSample2x(),
                nn.Conv2d(64, 32, kernel_size=1),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )

        elif model_type == "len_pred":
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Linear(512, 512)

            image_feature_dim = 512

            self.fc_final = nn.Sequential(
                nn.Linear(image_feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        elif model_type == "len_pred_new":
            self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Linear(512, 512)

            image_feature_dim = 512

            self.fc_feature = nn.Sequential(
                nn.Linear(num_classes, 32),
                nn.GELU(),
                nn.Linear(32, num_classes),
                nn.GELU(),
                nn.Dropout(0.1),
            )

            self.fc_final = nn.Sequential(
                nn.Linear(image_feature_dim + num_classes, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, image, feature=None):
        if self.model_type == "mask_pred":
            # Encoder (Downsampling)
            input = image / 255.0
            e1 = self.conv1(input)  # (750, 900) -> (375, 450)
            e2 = self.enc1(e1)  # (375, 450) -> (188, 225)
            e3 = self.enc2(e2)  # (188, 225) -> (94, 113)
            e4 = self.enc3(e3)  # (94, 113) -> (47, 57)
            bottleneck_input = self.enc4(e4)  # (47, 57) -> (24, 29)

            # Bottleneck
            bottleneck_output = self.bottleneck(bottleneck_input)

            # Decoder (Upsampling)
            d4 = self.dec4(bottleneck_output)

            cat4 = torch.cat((d4, e4), dim=1)
            d3 = self.dec3(cat4)

            cat3 = torch.cat((d3, e3), dim=1)
            d2 = self.dec2(cat3)

            cat2 = torch.cat((d2, e2), dim=1)
            d1 = self.dec1(cat2)

            cat1 = torch.cat((d1, e1), dim=1)
            output = self.final_conv(cat1)

            return output

        elif self.model_type == "len_pred":
            mask_features = self.resnet(image)

            output = self.fc_final(mask_features)  # [176, 1]
            return output

        elif self.model_type == "len_pred_new":
            mask_features = self.resnet(image)

            non_image_features = self.fc_feature(feature)  # [176, 64]

            combined = torch.cat((mask_features, non_image_features), dim=1)  # [176, 576]

            output = self.fc_final(combined)  # [176, 1]
            return output


class UpSample2x(nn.Module):
    def __init__(self):
        super(UpSample2x, self).__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x):
        input_shape = list(x.shape)
        # unsqueeze is expand_dims equivalent
        # permute is transpose equivalent
        # view is reshape equivalent
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        ret = ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))
        return ret


class DenseBlock(nn.Module):
    def __init__(self, in_ch, unit_count):
        super(DenseBlock, self).__init__()

        self.nr_unit = unit_count

        unit_in_ch = in_ch
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(unit_in_ch, 128, kernel_size=1, bias=False),

                    nn.BatchNorm2d(128, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 32, kernel_size=5, padding=2, bias=False, groups=4),
                )
            )
            unit_in_ch += 32

        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(unit_in_ch, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat
