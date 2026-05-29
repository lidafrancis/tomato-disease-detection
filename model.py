# model.py
import torch
import torch.nn as nn
from torchvision import models
from dataset_loader import NUM_CLASSES


def build_model(num_classes=NUM_CLASSES):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace only final layer — simple and clean
    model.fc = nn.Linear(512, num_classes)

    return model


if __name__ == "__main__":
    from torchinfo import summary
    model = build_model()
    print("ResNet18 Transfer Learning Model")
    print("=" * 60)
    summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params"],
        depth=2
    )
    dummy = torch.zeros(4, 3, 224, 224)
    output = model(dummy)
    print(f"\nForward pass:")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {output.shape}")
    print(f"\nModel OK!")