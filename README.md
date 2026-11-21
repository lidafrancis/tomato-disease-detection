# 🍅 Tomato Disease Detection

**A ResNet18 transfer-learning image classifier that detects tomato leaf diseases (PyTorch)**

This project classifies tomato leaf images into three categories: Early Blight, Late Blight and Healthy.

## 🌟 Core Idea

Instead of training a deep network from scratch, the project starts from a **ResNet18 pretrained on ImageNet**, freezes the entire backbone, and trains only a new final layer (512 → 3 classes) on tomato leaf images. This keeps training lightweight and fast, even on a CPU.

## 🍃 Classes

- Early Blight
- Late Blight
- Healthy

## 🔄 How It Works

1. **Data loading**: A custom PyTorch `Dataset` reads images from class-named folders (`td/tomato/train` and `td/tomato/val`). Class names are taken from the folder names.
2. **Preprocessing**: Images are resized to 224×224 and normalized with ImageNet statistics.
3. **Augmentation (training only)**: Random horizontal flip, vertical flip, rotation (±15°) and color jitter (brightness, contrast, saturation).
4. **Model**: Pretrained ResNet18 with all backbone layers frozen and a new `Linear(512, 3)` classifier head.
5. **Training**: Adam optimizer (lr = 0.001, weight decay = 1e-4) with cross-entropy loss, batch size 64, up to 10 epochs.
6. **Learning-rate scheduling**: `ReduceLROnPlateau` halves the learning rate when validation loss stops improving.
7. **Early stopping**: Training stops after 5 epochs with no improvement in validation loss.
8. **Checkpointing**: The best model (lowest validation loss) is saved to `best_model.pth`.
9. **Visualization**: Loss and accuracy curves are saved to `loss_curve.png`.

## 🛠 Tech Stack

- Python 3.13
- PyTorch
- torchvision
- Pillow
- Matplotlib
- torchinfo (optional, only used to print the model summary in `model.py`)

## ✨ Features

- Three-class tomato leaf disease classification
- Transfer learning with pretrained ResNet18
- Data augmentation to improve robustness
- Automatic best-model checkpointing
- Early stopping and learning-rate scheduling
- Training and validation loss/accuracy curves
- Single-image prediction with confidence score and top-3 probabilities
- Quick sanity test that predicts one validation image per class

## 📁 Project Structure

```
├── td/
│   └── tomato/
│       ├── train/            # one sub-folder per class
│       └── val/              # one sub-folder per class
├── dataset_loader.py         # Dataset, transforms and DataLoaders
├── model.py                  # ResNet18 transfer-learning model
├── train.py                  # Training loop, early stopping, plots
├── predict.py                # Predict a single image
├── test.py                   # Sanity test: one validation image per class
├── best_model.pth            # Saved best model
└── loss_curve.png            # Training curves
```

## 💻 Installation

```bash
pip install torch torchvision pillow matplotlib torchinfo
```

## 🚀 Usage

```bash
python train.py                    # train the model
python predict.py <image_path>     # predict on a single image
python test.py                     # quick check: one validation image per class
```

`predict.py` prints the predicted class, the confidence score, and the top 3 class probabilities.
