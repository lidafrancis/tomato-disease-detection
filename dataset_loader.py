# dataset_loader.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

TRAIN_DIR = "td/tomato/train"
VAL_DIR   = "td/tomato/val"

# 64x64 — better detail than 32x32, still fast on CPU
IMG_SIZE = 224

CLASS_NAMES  = sorted([
    d.name for d in Path(TRAIN_DIR).iterdir() if d.is_dir()
])
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES  = len(CLASS_NAMES)


class TomatoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples   = []

        for class_name in CLASS_NAMES:
            folder = Path(root_dir) / class_name
            if not folder.exists():
                continue
            for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG"]:
                for img_path in folder.glob(ext):
                    self.samples.append(
                        (img_path, CLASS_TO_IDX[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# Stronger augmentation — helps healthy class recognition
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ),
])

train_dataset = TomatoDataset(TRAIN_DIR, transform=train_transform)
val_dataset   = TomatoDataset(VAL_DIR,   transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

if __name__ == "__main__":
    print(f"Classes ({NUM_CLASSES}):")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx} : {name}")

    print(f"\nTrain images : {len(train_dataset)}")
    print(f"Val images   : {len(val_dataset)}")

    images, labels = next(iter(train_loader))
    print(f"\nOne batch:")
    print(f"  images shape  : {images.shape}")
    print(f"  labels shape  : {labels.shape}")
    print(f"  sample labels : {labels[:5].tolist()}")
    print(f"  sample classes: {[CLASS_NAMES[l] for l in labels[:5].tolist()]}")
    print(f"\nDataLoader OK!")