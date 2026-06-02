from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json

import torchvision.models as models
import torch.nn as nn

app = FastAPI()

# ── Cure database ─────────────────────────────────────────────────────────────

CURE_INFO = {
    "Tomato___Early_blight": {
        "display_name": "Early blight",
        "is_diseased": True,
        "cause": "Caused by the fungus Alternaria solani. Spreads in warm, humid conditions.",
        "cures": [
            "Remove and destroy all infected leaves immediately to stop the fungus spreading.",
            "Apply a copper-based fungicide (e.g. Bordeaux mixture) or chlorothalonil every 7–10 days.",
            "Water at the base of the plant — avoid wetting the leaves.",
            "Improve air circulation by pruning crowded branches.",
            "Practice crop rotation — do not grow tomatoes in the same spot next season.",
        ]
    },
    "Tomato___Late_blight": {
        "display_name": "Late blight",
        "is_diseased": True,
        "cause": "Caused by Phytophthora infestans. Thrives in cool, wet weather and spreads rapidly.",
        "cures": [
            "Remove infected plants immediately — late blight spreads very fast and can destroy an entire crop.",
            "Apply a fungicide containing mancozeb, chlorothalonil, or copper hydroxide at first sign.",
            "Avoid overhead irrigation and water early in the day so leaves dry quickly.",
            "Destroy (do not compost) all infected plant material.",
            "Plant resistant tomato varieties in future seasons.",
        ]
    },
    "Tomato___healthy": {
        "display_name": "Healthy",
        "is_diseased": False,
        "cause": None,
        "cures": []
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────

print("Loading model...")
checkpoint = torch.load("best_model.pth", map_location="cpu")
class_names = checkpoint["class_names"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded. Classes: {class_names}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── HTML UI ───────────────────────────────────────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tomato Disease Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f5f0; min-height: 100vh; padding: 2rem 1rem; color: #1a1a18; }
  .app { max-width: 620px; margin: 0 auto; }
  h1 { font-size: 22px; font-weight: 500; text-align: center; margin-bottom: 4px; }
  .subtitle { font-size: 14px; color: #666; text-align: center; margin-bottom: 1.5rem; }

  /* Upload zone */
  .upload-zone { background: #fff; border: 2px dashed #ccc; border-radius: 12px;
                 padding: 2.5rem 1rem; text-align: center; cursor: pointer;
                 transition: border-color 0.2s; }
  .upload-zone:hover, .upload-zone.drag-over { border-color: #888; }
  .upload-icon { font-size: 40px; margin-bottom: 8px; }
  .upload-zone p { font-size: 14px; color: #666; margin-bottom: 12px; }
  .choose-btn { display: inline-block; padding: 8px 20px; border: 1px solid #bbb;
                border-radius: 8px; font-size: 13px; cursor: pointer;
                background: #f5f5f0; }
  #file-input { display: none; }
  .preview-img { max-width: 100%; max-height: 200px; border-radius: 8px;
                 margin-bottom: 12px; object-fit: contain; }

  /* Spinner */
  .spinner { width: 32px; height: 32px; border: 3px solid #eee;
             border-top-color: #888; border-radius: 50%;
             animation: spin 0.8s linear infinite; margin: 1rem auto; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Result card */
  .result-card { background: #fff; border: 1px solid #e8e8e4;
                 border-radius: 12px; padding: 1.25rem; margin-top: 1rem; }
  .result-header { display: flex; align-items: center; gap: 10px; margin-bottom: 1rem; }
  .badge { font-size: 12px; font-weight: 500; padding: 3px 10px;
           border-radius: 6px; }
  .badge-diseased { background: #FAECE7; color: #993C1D; }
  .badge-healthy { background: #EAF3DE; color: #3B6D11; }
  .class-label { font-size: 17px; font-weight: 500; }

  /* Confidence bar */
  .conf-row { display: flex; justify-content: space-between;
              font-size: 13px; color: #666; margin-bottom: 4px; }
  .bar-bg { background: #f0f0ec; border-radius: 6px; height: 10px; margin-bottom: 1rem; }
  .bar-fill { height: 10px; border-radius: 6px;
              transition: width 0.6s ease; }
  .fill-disease { background: #D85A30; }
  .fill-healthy { background: #639922; }

  /* Cure section */
  .cure-section { border-top: 1px solid #eee; padding-top: 1rem; margin-top: 0.5rem; }
  .cure-section h3 { font-size: 15px; font-weight: 500; margin-bottom: 4px; }
  .cause-text { font-size: 13px; color: #888; margin-bottom: 12px; }
  .cure-item { display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start; }
  .cure-num { width: 22px; height: 22px; background: #FAECE7; color: #993C1D;
              border-radius: 50%; font-size: 11px; font-weight: 500;
              display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .cure-num-healthy { background: #EAF3DE; color: #3B6D11; }
  .cure-item p { font-size: 13px; color: #444; line-height: 1.5; margin-top: 2px; }

  /* All probabilities */
  .all-probs { border-top: 1px solid #eee; padding-top: 1rem; margin-top: 1rem; }
  .all-probs h3 { font-size: 12px; color: #888; font-weight: 500;
                  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px; }
  .prob-row { margin-bottom: 8px; }
  .prob-label { display: flex; justify-content: space-between;
                font-size: 13px; color: #555; margin-bottom: 3px; }
  .mini-bg { background: #f0f0ec; border-radius: 4px; height: 6px; }
  .mini-fill { height: 6px; border-radius: 4px; transition: width 0.6s ease; }

  /* Happy leaf */
  .healthy-msg { text-align: center; padding: 1rem 0; }
  .healthy-msg .big { font-size: 36px; margin-bottom: 8px; }
  .healthy-msg p { font-size: 14px; color: #3B6D11; }

  .error-msg { color: #993C1D; font-size: 14px; text-align: center; padding: 1rem; }
  #result { display: none; }
  #loading { display: none; text-align: center; padding: 1rem; font-size: 14px; color: #888; }
</style>
</head>
<body>
<div class="app">
  <h1>🍅 Tomato leaf disease detector</h1>
  <p class="subtitle">Upload a leaf photo to identify diseases and get treatment guidance</p>

  <div class="upload-zone" id="drop-zone">
    <div class="upload-icon">🌿</div>
    <img id="preview" class="preview-img" style="display:none" alt="Uploaded leaf">
    <p id="upload-hint">Drag and drop a leaf image here, or</p>
    <label class="choose-btn" for="file-input">Choose file</label>
    <input type="file" id="file-input" accept="image/*">
  </div>

  <div id="loading">
    <div class="spinner"></div>
    <p>Analysing your leaf...</p>
  </div>

  <div id="result"></div>
</div>

<script>
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const loading = document.getElementById('loading');
const resultDiv = document.getElementById('result');

function showLoading() {
  loading.style.display = 'block';
  resultDiv.style.display = 'none';
}

function hideLoading() {
  loading.style.display = 'none';
}

async function analyse(file) {
  showLoading();

  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    document.getElementById('upload-hint').textContent = file.name;
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: formData });
    if (!resp.ok) throw new Error('API error ' + resp.status);
    const data = await resp.json();
    hideLoading();
    renderResult(data);
  } catch (err) {
    hideLoading();
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<div class="result-card"><p class="error-msg">Error: ' + err.message + '</p></div>';
  }
}

function renderResult(data) {
  const cls = data.predicted_class;
  const conf = (data.confidence * 100).toFixed(1);
  const probs = data.all_probabilities;
  const info = CURE_INFO[cls];

  const isDisease = info.is_diseased;
  const badgeClass = isDisease ? 'badge-diseased' : 'badge-healthy';
  const badgeText = isDisease ? 'Diseased' : 'Healthy';
  const fillClass = isDisease ? 'fill-disease' : 'fill-healthy';

  let html = '<div class="result-card">';
  html += '<div class="result-header">';
  html += '<span class="badge ' + badgeClass + '">' + badgeText + '</span>';
  html += '<span class="class-label">' + info.display_name + ' detected</span>';
  html += '</div>';

  html += '<div class="conf-row"><span>Confidence</span><span>' + conf + '%</span></div>';
  html += '<div class="bar-bg"><div class="bar-fill ' + fillClass + '" style="width:' + conf + '%"></div></div>';

  if (isDisease) {
    html += '<div class="cure-section">';
    html += '<h3>🌿 Treatment guide</h3>';
    html += '<p class="cause-text">' + info.cause + '</p>';
    info.cures.forEach((cure, i) => {
      html += '<div class="cure-item">';
      html += '<div class="cure-num">' + (i + 1) + '</div>';
      html += '<p>' + cure + '</p>';
      html += '</div>';
    });
    html += '</div>';
  } else {
    html += '<div class="cure-section healthy-msg">';
    html += '<div class="big">✅</div>';
    html += '<p>Your tomato leaf looks healthy! Keep up the good care.</p>';
    html += '</div>';
  }

  html += '<div class="all-probs"><h3>All class probabilities</h3>';
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  sorted.forEach(([label, prob]) => {
    const pct = (prob * 100).toFixed(1);
    const shortLabel = label.replace('Tomato___', '').replace(/_/g, ' ');
    const color = label === 'Tomato___healthy' ? '#639922' : '#D85A30';
    html += '<div class="prob-row">';
    html += '<div class="prob-label"><span>' + shortLabel + '</span><span>' + pct + '%</span></div>';
    html += '<div class="mini-bg"><div class="mini-fill" style="width:' + pct + '%;background:' + color + '"></div></div>';
    html += '</div>';
  });
  html += '</div></div>';

  resultDiv.innerHTML = html;
  resultDiv.style.display = 'block';
}

const CURE_INFO = """ + json.dumps(CURE_INFO) + """;

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) analyse(e.target.files[0]);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) analyse(e.dataTransfer.files[0]);
});
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_idx = probs.argmax().item()
    predicted_class = class_names[predicted_idx]
    confidence = probs[predicted_idx].item()
    all_probs = {class_names[i]: round(probs[i].item(), 4) for i in range(len(class_names))}

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probs,
    })