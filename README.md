# Vision-Transformers-for-Image-Forgery-and-Deepfake-Detection

# Deepfake vs Real Image Classifier (CNN + ViT on FaceForensics++)

This project builds a simple **deepfake-vs-real image classifier** using a subset of the **FaceForensics++** dataset.  
The goal is to detect manipulated facial videos by:

- extracting faces from video frames  
- preprocessing them  
- and training **CNN-based and ViT-based models**.

Because the full dataset is very large, we work with a **smaller subset** (e.g., a few hundred real and a few hundred fake videos).  
This makes it practical to experiment without needing huge compute.

---

## Overview

The core pipeline is:

1. Take real and fake videos  
2. Break them into frames  
3. Detect and crop the face  
4. Resize the images  
5. Train a classifier to tell **real vs fake**

This setup focuses on the **most informative region** (the face) and keeps the problem tractable on limited hardware.

---

## Preprocessing

### 1. Face Detection and Cropping

- We use **dlib’s face detector** to locate the face in each frame.  
- The image is then **cropped around the detected face**, so the model focuses on the main region that matters and ignores background clutter.

### 2. Resizing

- All cropped faces are resized to **224 × 224** pixels.  
- This resolution matches the input size required by:
  - most **CNN backbones** (e.g., Xception, EfficientNet, MobileNetV3)  
  - and common **Vision Transformer (ViT) backbones**.

### 3. Pickled Dataset

- After preprocessing, the **images and labels** are saved into a **pickle file**.  
- This:
  - avoids re-processing every time  
  - ensures **all models** (CNNs and ViTs) train on the **exact same data**, making comparisons fair and reproducible.

---

## Model

The training script allows switching between different **backbones**, such as:

- **CNNs:** Xception, EfficientNet, MobileNetV3  
- **ViTs:** standard ViT-based backbones (e.g., ViT-Base–style architectures)

Each backbone is followed by a **simple classification head** for **binary classification** (real vs fake).

You can:

- **Freeze** backbone layers for transfer learning on a small dataset  
- **Unfreeze** some or all layers to **fine-tune** the backbone end-to-end, depending on your training strategy.

---

> **Note:**  
> Edit the **configuration file/arguments** (e.g., backbone choice, paths, training hyperparameters) **before running the code**.

### Data: https://drive.google.com/drive/folders/1ZLOIVmzC0MBYkmjJOdWJKqNfeKon7QI2?usp=sharing
