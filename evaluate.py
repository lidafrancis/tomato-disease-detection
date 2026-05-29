# evaluate.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import build_model
import numpy as np

DEVICE = 'cpu'
checkpoint = torch.load('best_model.pth', map_location=DEVICE)
CLASS_NAMES = checkpoint['class_names']
model = build_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Collect all predictions and true labels
all_preds = []
all_labels = []

VAL_DIR = Path('td/tomato/val')

print("Running evaluation on validation set...")
print("-" * 50)

for class_folder in sorted(VAL_DIR.iterdir()):
    if not class_folder.is_dir():
        continue
    true_class = class_folder.name
    if true_class not in CLASS_NAMES:
        continue
    true_idx = CLASS_NAMES.index(true_class)

    images = (list(class_folder.glob('*.jpg')) +
              list(class_folder.glob('*.JPG')) +
              list(class_folder.glob('*.png')))

    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
        pred_idx = probs.argmax().item()
        all_preds.append(pred_idx)
        all_labels.append(true_idx)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
num_classes = len(CLASS_NAMES)

# ---- Confusion Matrix ----
print("\nConfusion Matrix:")
print("(Rows = True class, Columns = Predicted class)")
print()

header = f"{'':35}" + "".join(f"{name[:12]:>15}" for name in CLASS_NAMES)
print(header)
print("-" * (35 + 15 * num_classes))

confusion = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(all_labels, all_preds):
    confusion[true][pred] += 1

for i, name in enumerate(CLASS_NAMES):
    row = f"{name[:35]:<35}" + "".join(f"{confusion[i][j]:>15}" for j in range(num_classes))
    print(row)

# ---- Per class metrics ----
print("\n" + "=" * 70)
print("Per-Class Metrics:")
print("=" * 70)
print(f"{'Class':<35} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'Accuracy':>10}")
print("-" * 70)

precisions, recalls, f1s = [], [], []

for i, name in enumerate(CLASS_NAMES):
    tp = confusion[i][i]
    fp = confusion[:, i].sum() - tp
    fn = confusion[i, :].sum() - tp
    tn = confusion.sum() - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = tp / confusion[i].sum() * 100 if confusion[i].sum() > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    print(f"{name[:35]:<35} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {accuracy:>9.1f}%")

print("-" * 70)
print(f"{'Macro Average':<35} {np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f} {np.mean(f1s):>10.3f}")

# ---- Overall accuracy ----
overall = (all_preds == all_labels).sum() / len(all_labels) * 100
print(f"\nOverall Accuracy : {overall:.1f}%")
print(f"Total Images     : {len(all_labels)}")
print("=" * 70)