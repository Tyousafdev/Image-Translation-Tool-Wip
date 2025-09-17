import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

# --- Setup ---
device = "mps" if torch.backends.mps.is_available() else "cpu"

# same transform you use in server.py
transform = T.Compose([
    T.Grayscale(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# dynamically get class order (bubble/sfx)
class_names = sorted([d.name for d in Path("dataset/train").iterdir() if d.is_dir()])

# --- Load model ---
model = models.resnet18(weights=None, num_classes=2)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
state = torch.load("bubble_classifier.pth", map_location=device)
missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print("⚠️ State_dict mismatch!")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
model.to(device).eval()

# --- Pick test image ---
img_path = input("Path to a crop image: ").strip()
img = Image.open(img_path).convert("L")
x = transform(img).unsqueeze(0).to(device)

# --- Predict ---
with torch.no_grad():
    out = model(x)
    probs = torch.softmax(out, dim=1)[0]

for i, p in enumerate(probs):
    print(f"{class_names[i]}: {p.item():.4f}")

pred_idx = torch.argmax(probs).item()
print("✅ Predicted:", class_names[pred_idx])
