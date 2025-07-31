# 🚗 Car Model Identification with Bilinear CNN

This project implements a **fine-grained image classification** system to identify car models from images. The goal is to achieve over 65% accuracy, and we reached **71.94%** test accuracy using a **Bilinear CNN** architecture with pretrained models. All code is implemented in **PyTorch**.

## 📁 Project Structure

```bash
car_model_identification/
├── data/                      # .npy files with train/test data
│   ├── x_train.npy
│   ├── y_train.npy
│   ├── x_test.npy
│   └── y_test.npy
├── models/                    # Model architecture
│   └── bilinear_cnn.py
├── datasets/                  # Custom PyTorch dataset
│   └── car_dataset.py
├── transforms/                # Data augmentation pipeline
│   └── data_transforms.py
├── utils/                     # Training and evaluation utilities
│   └── train_eval.py
├── main.py                    # Main script: train and evaluate the model
├── config.py                  # Experiment configuration
├── requirements.txt           # Environment dependencies
└── README.md                  
```

## ⚙️ What does this project do?

- Loads and preprocesses a dataset of car images labeled into 20 fine-grained categories.
- Applies data augmentation techniques to improve generalization.
- Defines a **Bilinear Convolutional Neural Network (Bilinear CNN)** combining two pretrained models.
- Trains the model using PyTorch and evaluates it on a separate test set.
- Saves the trained model and visualizes example predictions.

## 🧠 Model Architecture

The model uses a bilinear structure composed of two pretrained convolutional backbones:

- **ResNet-50** (without its classification head).
- **VGG-16** (convolutional layers only).

Both networks extract feature maps from the same input image. These are combined using an **outer product bilinear pooling** operation to capture pairwise feature interactions. The result is passed through a linear layer for classification.

> 🔁 This bilinear pooling technique is commonly used in **fine-grained visual recognition** tasks, where subtle differences between classes are important.

## 📊 Results

- **Final test accuracy:** `71.94%`
- **Number of classes:** 20
- **Epochs:** 10
- **Learning rate:** 0.0001
- **GPU support:** ✅ (automatically used if available)

## 🚀 Requirements

```bash
pip install -r requirements.txt
```