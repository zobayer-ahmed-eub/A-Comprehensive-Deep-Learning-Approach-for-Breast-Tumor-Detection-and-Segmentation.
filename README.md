Breast Cancer Ultrasound Analysis: Classification & Segmentation
This repository contains the implementation of an end-to-end automated system for the detection, classification, and segmentation of breast lesions in ultrasound images. The project utilizes deep learning techniques—specifically Convolutional Neural Networks (CNNs) and Transformers—to assist clinicians in early breast cancer detection.

Project Overview
The system performs two primary parallel tasks:

Classification: Categorizes breast lesions into three classes: Normal, Benign, and Malignant.

Segmentation: Delineates the precise boundaries of lesions to aid in analyzing size and shape.

The framework integrates advanced techniques such as Squeeze-and-Excitation (SE) blocks, Focal Loss for class imbalance, and ensemble boosting to enhance performance.

Repository Structure
The codebase is divided into classification and segmentation modules, featuring implementations in both PyTorch and TensorFlow/Keras.

1. Classification Models
These notebooks implement various backbone architectures to classify ultrasound images.

ConvNeXt Tiny (PyTorch)

convonext-tiny-base.ipynb: Base implementation of the ConvNeXt Tiny model.

convonext-tiny-mod.ipynb: Modified ConvNeXt Tiny implementation, likely including hyperparameter tuning or architectural tweaks.

EfficientNet B0

efficientnet-b0.ipynb: TensorFlow/Keras implementation of EfficientNet B0.

efficientnet-b0-mod.ipynb: PyTorch implementation of EfficientNet B0.

ResNet-50

resnet-50.ipynb: TensorFlow/Keras implementation of ResNet-50.

resnet-50-mod.ipynb: PyTorch implementation of ResNet-50.

resnet-50-se-class-focalloss.ipynb: A specialized TensorFlow implementation integrating Squeeze-and-Excitation (SE) Attention blocks and Focal Loss to handle difficult samples and class imbalance.

2. Segmentation Models
These notebooks focus on generating pixel-level masks for lesion detection using TensorFlow/Keras.

breast_cancer_image_segmentation_with_Unet.ipynb: Standard U-Net architecture implementation for biomedical image segmentation.

breast-cancer-image-segmentation-Resunet.ipynb: Implementation of ResUNet, a hybrid architecture combining ResNet residual blocks with the U-Net structure for improved gradient flow and feature extraction.

Key Features & Techniques
Deep Learning Frameworks: Implementations in both PyTorch and TensorFlow.

Data Augmentation: Usage of Albumentations for robust training (e.g., rotations, flips, brightness adjustments).

Attention Mechanisms: Integration of SE-Blocks to focus on channel-wise feature interdependencies.

Loss Functions: Implementation of Focal Loss and Weighted Categorical Crossentropy to address class imbalance in medical datasets.

Evaluation Metrics: Comprehensive tracking of Accuracy, Precision, Recall, F1-Score, AUC-ROC, IoU (Intersection over Union), and Dice Coefficient.

Getting Started
Prerequisites
Python 3.x

Jupyter Notebook / Google Colab / Kaggle Kernels

Libraries:

torch, torchvision

tensorflow, keras

scikit-learn

albumentations

matplotlib, seaborn, numpy, pandas

Dataset
The models are designed to work with breast ultrasound datasets (e.g., Breast Ultrasound Images Dataset - BUSI). Ensure the data is structured with separate folders for images and masks or classified subfolders (Normal, Benign, Malignant) before running the notebooks.

Usage
Clone the repository.

Install dependencies.

Open the desired notebook (e.g., resnet-50-mod.ipynb for classification or breast_cancer_image_segmentation_with_Unet.ipynb for segmentation).

Update the DATA_DIR paths to point to your local dataset location.

Run the cells to train the model and visualize results.
