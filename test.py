# test.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import TomatoCNN

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"

checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = checkpoint['class_names']
model       = TomatoCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Must match training image size
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict(image_path):
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    top3_probs, top3_idxs = probs.topk(3)
    print(f"\n  File       : {Path(image_path).name[:50]}")
    print(f"  Prediction : {CLASS_NAMES[top3_idxs[0]]}")
    print(f"  Confidence : {top3_probs[0]*100:.1f}%")
    print(f"  Top 3:")
    for prob, idx in zip(top3_probs, top3_idxs):
        bar = "█" * int(prob * 30)
        print(f"    {prob*100:5.1f}%  {CLASS_NAMES[idx]:<45}  {bar}")
    print("-" * 60)

VAL_DIR = Path("td/tomato/val")
print("=" * 60)
print("  Testing one image from each class")
print("=" * 60)

for class_folder in sorted(VAL_DIR.iterdir()):
    if not class_folder.is_dir():
        continue
    images = list(class_folder.glob("*.jpg")) + \
             list(class_folder.glob("*.JPG")) + \
             list(class_folder.glob("*.png"))
    if images:
        predict(images[0])