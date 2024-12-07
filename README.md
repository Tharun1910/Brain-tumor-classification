# Brain Tumor Classification using Deep Learning 

This repository contains the implementation of a deep learning-based approach to classify brain tumors using MRI images. The project leverages pre-trained models like VGG19, DenseNet-121, and MobileNetV2, and uses a Random Forest classifier for enhanced performance.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)


---

## Overview
This project focuses on classifying brain tumors into four categories: 
1. Glioma
2. Meningioma
3. Pituitary Tumor
4. No Tumor

The aim is to automate the diagnostic process by utilizing transfer learning on MRI images and enhancing results with feature-based Random Forest classification.

---

## Dataset
The dataset used in this project is a combination of:
- Figshare
- SARTAJ
- Br35H
-The combined Dataset : [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The dataset contains **7,023 MRI images** categorized into the four mentioned classes. Preprocessing and augmentation techniques were applied to improve the model's generalization.

---

## Models
The following models were used:
1. **Pre-trained Models**:
   - VGG19
   - DenseNet-121
   - MobileNetV2
2. **Custom DenseNet-121**:
   - Fine-tuned with additional dropout layers and reduced learning rates.
3. **Random Forest Classifier**:
   - Trained on features extracted from the custom DenseNet-121.

### Results:
| Model           | Accuracy   |
|------------------|------------|
| VGG19           | 85.4%      |
| MobileNetV2     | 89.3%      |
| DenseNet-121    | 91.76%     |
| Fine-tuned DenseNet-121 | 98.5% |
| Random Forest (Feature-based) | 100% |

---

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
