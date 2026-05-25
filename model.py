# model.py
import torch
import torch.nn as nn
from dataset_loader import NUM_CLASSES


class TomatoCNN(nn.Module):
    """
    4 conv blocks for 64x64 input:
    64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
    Final: 256 channels x 4x4 = 4096 values
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(TomatoCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                         # 256x4x4 = 4096
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from torchinfo import summary

    model = TomatoCNN()

    print("Model Architecture Summary")
    print("=" * 60)
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params"],
        depth=2
    )

    dummy  = torch.zeros(4, 3, 64, 64)
    output = model(dummy)
    print(f"\nForward pass test:")
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {output.shape}")
    print(f"\nModel OK!")