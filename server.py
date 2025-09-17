import os, io, cv2
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import numpy as np
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from manga_ocr import MangaOcr

# ---------- Setup ----------
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
ocr = MangaOcr()

# Load Hugging Face RT-DETR model (bubble + text detection)
model_path = hf_hub_download("ogkalu/comic-text-and-bubble-detector", "model.onnx")
session = ort.InferenceSession(model_path)

# ---------- Helpers ----------
def preprocess(np_img, size=640):
    """Resize + normalize image for RT-DETR"""
    h, w = np_img.shape[:2]
    img = cv2.resize(np_img, (size, size))
    img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return img, (h, w)

def detect_bubbles_and_text(np_img):
    """Run RT-DETR detection"""
    inp, (orig_h, orig_w) = preprocess(np_img)
    outputs = session.run(None, {"images": inp})[0]

    results = []
    for det in outputs:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.3:  # confidence threshold
            continue
        # Scale back to original image size
        x1 = int(x1 / 640 * orig_w)
        y1 = int(y1 / 640 * orig_h)
        x2 = int(x2 / 640 * orig_w)
        y2 = int(y2 / 640 * orig_h)

        results.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "conf": float(conf),
            "class": int(cls)  # 0=bubble, 1=text_bubble, 2=text_free
        })
    return results

def draw_debug(np_img, detections, out_path="debug_balloons.png"):
    dbg = np_img.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        cls = det["class"]
        color = (0, 255, 0) if cls == 1 else ((0, 0, 255) if cls == 2 else (255, 0, 0))
        cv2.rectangle(dbg, (x, y), (x+w, y+h), color, 2)
    cv2.imwrite(out_path, dbg)
    return out_path

# ---------- API ----------
app = FastAPI()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    np_img = np.array(pil_img)

    # Step 1: detect bubbles + text
    detections = detect_bubbles_and_text(np_img)
    debug_path = draw_debug(np_img, detections)

    results = []
    for det in detections:
        x, y, w, h = det["bbox"]
        cls = det["class"]

        crop = pil_img.crop((x, y, x+w, y+h))

        if cls == 2:  # text outside bubbles (SFX)
            results.append({
                "bbox": det["bbox"],
                "text": "",
                "type": "sfx",
                "conf": det["conf"]
            })
            continue

        if cls == 1:  # text inside bubbles
            text = ocr(crop).strip()
            results.append({
                "bbox": det["bbox"],
                "text": text,
                "type": "dialogue",
                "conf": det["conf"]
            })

    return JSONResponse({
        "results": results,
        "debug_image": debug_path
    })
