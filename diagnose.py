import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path
from model import build_model

checkpoint = torch.load('best_model.pth', map_location='cpu')
CLASS_NAMES = checkpoint['class_names']
print(f'Classes : {CLASS_NAMES}')
print(f'Val acc : {checkpoint["val_acc"]}')
print(f'Epoch   : {checkpoint["epoch"]}')

model = build_model(num_classes=len(CLASS_NAMES))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print('\nRaw probabilities for one image from each class:')
print('-' * 60)

for folder in ['Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy']:
    images = list(Path(f'td/tomato/val/{folder}').glob('*.jpg'))
    if not images:
        print(f'{folder}: no images found')
        continue
    img = Image.open(images[0]).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
    pred = CLASS_NAMES[probs.argmax().item()]
    print(f'\nTrue class : {folder}')
    print(f'Predicted  : {pred}')
    for name, prob in zip(CLASS_NAMES, probs.tolist()):
        bar = '█' * int(prob * 30)
        print(f'  {name:<35} {prob*100:5.1f}%  {bar}')