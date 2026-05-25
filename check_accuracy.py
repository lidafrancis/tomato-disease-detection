import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import TomatoCNN

DEVICE = 'cpu'
checkpoint = torch.load('best_model.pth', map_location=DEVICE)
CLASS_NAMES = checkpoint['class_names']
model = TomatoCNN(num_classes=len(CLASS_NAMES))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_DIR = Path('td/tomato/val')
results = {name: {'correct': 0, 'total': 0} for name in CLASS_NAMES}

for class_folder in VAL_DIR.iterdir():
    if not class_folder.is_dir():
        continue
    true_class = class_folder.name
    if true_class not in CLASS_NAMES:
        continue
    images = (list(class_folder.glob('*.jpg')) +
              list(class_folder.glob('*.JPG')) +
              list(class_folder.glob('*.png')))
    for img_path in images:
        image = Image.open(img_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor), dim=1)[0]
        pred = CLASS_NAMES[probs.argmax().item()]
        results[true_class]['total'] += 1
        if pred == true_class:
            results[true_class]['correct'] += 1

print('\nPer-class accuracy:')
for name, r in results.items():
    acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
    print(f'  {name}: {r["correct"]}/{r["total"]} = {acc:.1f}%')