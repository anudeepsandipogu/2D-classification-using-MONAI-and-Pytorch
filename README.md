# 2D-classification-using-MONAI-and-Pytorch

2D Medical Image Classification with MONAI & PyTorch

This repository implements a 2D medical image classification pipeline using the MedNIST dataset
. The project demonstrates how MONAI can be applied for image preprocessing, augmentation, training, and evaluation in medical imaging tasks.

Features

Dataset Setup

Automatic download and setup of the MedNIST dataset (6 classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT).

Preprocessing & Augmentation

MONAI transforms: LoadImage, EnsureChannelFirst, ScaleIntensity.

Randomized augmentations: RandFlip, RandRotate, RandZoom.

Model Training

DenseNet121 backbone for robust 2D classification.

Loss: Cross-Entropy.

Optimizer: Adam.

Evaluation

Metrics: Accuracy, ROC-AUC.

Confusion matrix and ROC visualization.

Tech Stack

Frameworks: MONAI, PyTorch, Torchvision

Languages: Python

Tools: Jupyter Notebook, CUDA

Dataset

Source: MedNIST dataset

Classes: AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT

Task: Supervised classification of medical images
