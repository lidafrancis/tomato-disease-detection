# Tomato Disease Detection

A CNN-based image classifier that detects tomato leaf diseases.

## Classes
- Early Blight
- Late Blight
- Healthy

## How to run
```bash
  python train.py        # train the model
  python predict.py <image_path>  # predict on a single image
  python test.py         # test on val set
```

## Tech stack
- Python 3.13
- PyTorch
- torchvision