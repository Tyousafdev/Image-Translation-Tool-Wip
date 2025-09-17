import os, cv2, torch
import numpy as np
from pathlib import Path
from PIL import Image
from craft_pytorch.craft import CRAFT
from craft_pytorch import imgproc, craft_utils

# --- Init model ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
craft_net = CRAFT()
weights_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "craft_mlt_25k.pth"
state = torch.load(weights_path, map_location=device, weights_only=False)
if "state_dict" in state:
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
craft_net.load_state_dict(state, strict=False)
craft_net.to(device).eval()

# --- Settings ---
CRAFT_CANVAS = 1280
CRAFT_TEXT_THR = 0.7
CRAFT_LINK_THR = 0.4
CRAFT_LOW_THR = 0.4
MIN_AREA = 300

# --- Helpers ---
def detect_text_polygons(image: np.ndarray):
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, CRAFT_CANVAS, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1.0 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        y, _ = craft_net(x)
    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()
    boxes, _ = craft_utils.getDetBoxes(
        score_text, score_link,
        text_threshold=CRAFT_TEXT_THR,
        link_threshold=CRAFT_LINK_THR,
        low_text=CRAFT_LOW_THR
    )
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes

def polys_to_rects(polys):
    rects = []
    for b in polys:
        pts = np.array(b).astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        if w * h >= MIN_AREA:
            rects.append([int(x), int(y), int(x+w), int(y+h)])
    return rects

def get_start_index(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir) if f.startswith("crop_") and f.endswith(".png")]
    if not existing:
        return 0
    nums = [int(f[5:-4]) for f in existing]
    return max(nums) + 1

def export_crops_from_image(image_path, out_dir, counter):
    img = Image.open(image_path).convert("RGB")
    np_img = np.array(img)
    rects = polys_to_rects(detect_text_polygons(np_img))
    saved = 0
    for (x1, y1, x2, y2) in rects:
        crop = img.crop((x1, y1, x2, y2))
        crop.save(os.path.join(out_dir, f"crop_{counter+saved:05d}.png"))
        saved += 1
    return saved

def export_crops_from_folder(folder, out_dir="dataset/unlabeled"):
    os.makedirs(out_dir, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg"}
    counter = get_start_index(out_dir)
    for root, _, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                img_path = os.path.join(root, fname)
                n = export_crops_from_image(img_path, out_dir, counter)
                print(f"✅ {fname}: saved {n} crops")
                counter += n
    print(f"\n✅ All done. Total crops in folder now: {len(os.listdir(out_dir))}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python export_crops.py /path/to/folder")
    else:
        export_crops_from_folder(sys.argv[1])
