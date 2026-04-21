# FishLEN
FishLEN(Fish Length Estimation Network): A deep learning framework for estimating fish length from images, developed for research in the Dormant Biology Lab at Stony Brook University.

# Steps

## Step 1 Problem Analysis & Overall Strategy

To predict the fish length, we designed a **two-stage pipeline**. First, we segment the fish from the background to obtain a clean binary mask (0 = fish region, 1 = background). Second, we use this binary mask as the input for a regression model to estimate fish length. This two-step design isolates irrelevant background information and ensures that the downstream model focuses solely on fish morphology.



## Step 2 Model Selection & Architecture Design

### 2.1 Segmentation Model: U-Net + ResNet34 Encoder

For the segmentation stage, we chose **U-Net with a ResNet34 encoder** because U-Net’s U-shaped structure effectively preserves spatial details through skip connections, making it highly suitable for biomedical-style images such as fish silhouettes. The ResNet34 encoder further strengthens feature extraction with residual learning, providing stable and expressive representations even when the training set is small. This combination offers a good balance between accuracy and computational efficiency while remaining easy to fine-tune for our task.



### 2.2 Length Prediction Model: ResNet18 + MLP

For length prediction, we use **ResNet18** to extract morphological features directly from the binary mask. Since the mask already removes background complexity, a lightweight encoder is sufficient to capture fish shape information. The extracted features are then passed through a simple 3-layer MLP to model the nonlinear relationship between the mask’s geometric structure and the actual fish length.



## Step 3 Training Configuration

We use **Adam** as the optimizer.

- For segmentation: **Dice Loss** to handle class imbalance and emphasize boundary accuracy.
- For length prediction: **L1 Loss**, which is stable and produces interpretable error magnitudes. 



## Step 4 Ground Truth Generation via SAM

We used **SAM (Segment Anything Model)** to automatically generate candidate segmentation masks for each fish image. SAM produces binary masks for all detectable objects within an image, which means it often segments multiple regions rather than only the fish. Because of this behavior, we need to manually inspect all generated masks and select the one that correctly isolates the fish from the background. The selected mask is then treated as the ground truth for training the segmentation model. This semi-automatic process significantly reduces our manual labeling effort while still ensuring high-quality segmentation labels suitable for supervised learning.



## Step 5 Image Preprocessing

Because the segmentation model requires an input resolution of **1920×1088**, while our original images are **1920×1080**, we need to apply padding to match the required dimensions. To achieve this, we add an **8-pixel padding region to the bottom** of both the image and the corresponding mask, using a pixel value of **0** for padding. This ensures that all inputs conform to the model’s expected size without altering the original content of the fish region, allowing consistent preprocessing across the entire dataset.



## Step 6 Dataset Structure

We need to design the structure of the dataset, here is the shape of it:

### Training dataset

```bash
dataset/
├── Day 1/
|	├── Image 1.png
|	├── Image 2.png
│   └── ...
├── Day 2/
|	├── Image 1.png
|	├── Image 2.png
│   └── ...
├── ...
└── dataset_infor.xlsx
```

### Testing dataset

```bash
dataset/
├── Day 1/
│   ├── Fish ID 1/
|	|	├── Image 1
|	|	├── Image 2
|	|	├── ...
│   ├── Fish ID 2/
|	|	├── Image 1
|	|	├── Image 2
|	|	├── ...
│   └── ...
├── Day 2/
│   ├── Fish ID 1/
|	|	├── Image 1
|	|	├── Image 2
|	|	├── ...
│   ├── Fish ID 2/
|	|	├── Image 1
|	|	├── Image 2
|	|	├── ...
│   └── ...
├── ...
└── dataset_infor.xlsx
```

### dataset_infor.xlsx

| Day Index | fish_no | sex  | avg_length_mm |              final_image              |
| :-------: | :-----: | :--: | :-----------: | :-----------------------------------: |
|    35     |    1    |  F   |   29.278855   | ,Screen Shot 2022-03-28 at 4.31.57 PM |
|    42     |    3    |  M   |  28.2284891   | Screen Shot 2022-04-04 at 3.26.37 PM  |
|    ...    |   ...   | ...  |      ...      |                  ...                  |

- Since we take an another analyst about the sex, we have a "sex" column in the excel file, you can drop that if you don't need it.



## Step 7 Training Subsampling Strategy

We believe that the current model architecture is sufficiently powerful to learn the relationship between fish shape and length even with a relatively small amount of training data. Therefore, we selected only **15% of the total images** from the dataset for training. To ensure fair and balanced sampling, each day’s folder contributes 15% of its images to the final training set. If 15% of a folder’s size is less than one image, we randomly select **one** image from that folder instead. Following this procedure, we obtained a total of **208 images**, which constitute the dataset used for training the model.

