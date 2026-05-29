# predict.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys
from model import build_model

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"

checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = checkpoint['class_names']
model = build_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
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
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0]

    top3_probs, top3_idxs = probs.topk(3)

    print("\n" + "=" * 50)
    print(f"  Image : {image_path}")
    print("=" * 50)
    print(f"  Prediction : {CLASS_NAMES[top3_idxs[0]]}")
    print(f"  Confidence : {top3_probs[0]*100:.1f}%")
    print("\n  Top 3:")
    for prob, idx in zip(top3_probs, top3_idxs):
        bar = "█" * int(prob * 40)
        print(f"  {CLASS_NAMES[idx]:<50} {prob*100:5.1f}%  {bar}")
    print("=" * 50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict(sys.argv[1])