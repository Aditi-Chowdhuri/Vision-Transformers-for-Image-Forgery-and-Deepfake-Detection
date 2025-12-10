# Vision-Transformers-for-Image-Forgery-and-Deepfake-Detection

# Deepfake vs Real Image Classifier (FaceForensics++ Subset)

This project builds a simple **deepfake-vs-real image classifier** using a subset of the **FaceForensics++** dataset.  
The goal is to detect manipulated facial videos by:

- extracting faces from video frames  
- preprocessing them  
- and training a **CNN-based model** for binary classification (real vs fake).

Because the full dataset is very large, we work with a **smaller subset** (e.g., a few hundred real and a few hundred fake videos).  
This makes it practical to experiment without needing huge compute.

---

## Overview

The core pipeline is:

1. Take real and fake videos  
2. Break them into frames  
3. Detect and crop the face  
4. Resize the images  
5. Train a classifier to tell real from fake  

This setup focuses on the **most informative region** (the face) and keeps the problem tractable on limited hardware.

---

## Preprocessing

### 1. Face Detection and Cropping

- Use **dlib’s face detector** to locate the face in each frame.  
- Crop around the detected face region so that:
  - the model focuses on the main area of interest
  - background noise is reduced.

### 2. Resizing

- All cropped faces are resized to **224 × 224** pixels.  
- This resolution is compatible with most standard CNN backbones such as:
  - **Xception**
  - **EfficientNet**
  - **MobileNetV3**

### 3. Pickled Dataset

- After preprocessing:
  - images (as tensors/arrays)  
  - and labels (real vs fake)  
  are saved into a **pickle file**.
- Benefits:
  - Avoids re-running expensive preprocessing every time.
  - Ensures **all models** train on the **exact same preprocessed data**, making comparisons fair.

---

## Model

- The training script supports **multiple CNN backbones**, including:
  - **Xception**
  - **EfficientNet**
  - **MobileNetV3**
- Each backbone is followed by a **simple classification head** for **binary classification** (real vs fake).

You can also:

- **Freeze** backbone layers for transfer learning with a small dataset  
- **Unfreeze** some or all layers to **fine-tune** the backbone end-to-end, depending on your training strategy.

### Data: https://drive.google.com/drive/folders/1ZLOIVmzC0MBYkmjJOdWJKqNfeKon7QI2?usp=sharing
