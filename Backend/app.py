# -*- coding: utf-8 -*-
"""
SulatAI — FastAPI Backend
=========================
Implements the full segmentation + ReXNet-150 inference pipeline
exactly as described in the capstone proposal:

  Step A : Receive uploaded multi-character image
  Step B : Grayscale → Adaptive Threshold → Contour Detection
           → Filter contours by area / aspect-ratio
  Step C : Crop each valid ROI, resize to 224×224, ImageNet-normalise
  Step D : Batch inference through ReXNet-150 (rexnet_baybayin_final.pth)
  Step E : Return JSON payload  {detections, annotated_image, stats}

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import base64
import time
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("sulatai")

# ---------------------------------------------------------------------------
# Constants — must match training notebook exactly
# ---------------------------------------------------------------------------
IMG_SIZE       = 224
IMAGENET_MEAN  = [0.485, 0.456, 0.406]
IMAGENET_STD   = [0.229, 0.224, 0.225]
MAX_FILE_BYTES = 10 * 1024 * 1024          # 10 MB upload limit
ALLOWED_TYPES  = {"image/png", "image/jpeg", "image/webp"}

# 59 Baybayin classes — identical ordering used during training
# ('d_r' was merged into 'd' before training, so only 59 unique labels)
BAYBAYIN_CLASSES: List[str] = [
    "a",       "b",       "ba",      "be_bi",   "bo_bu",
    "d",       "da",      "de_di",   "do_du",   "e_i",
    "g",       "ga",      "ge_gi",   "go_gu",   "h",
    "ha",      "he_hi",   "ho_hu",   "k",       "ka",
    "ke_ki",   "ko_ku",   "l",       "la",      "le_li",
    "lo_lu",   "m",       "ma",      "me_mi",   "mo_mu",
    "n",       "na",      "ne_ni",   "ng",      "nga",
    "nge_ngi", "ngo_ngu", "no_nu",   "o_u",     "p",
    "pa",      "pe_pi",   "po_pu",   "s",       "sa",
    "se_si",   "so_su",   "t",       "ta",      "te_ti",
    "to_tu",   "w",       "wa",      "we_wi",   "wo_wu",
    "y",       "ya",      "ye_yi",   "yo_yu",
]

# Friendly display mapping: model class label → human-readable transliteration
CLASS_DISPLAY: dict[str, str] = {
    "a":       "A",       "b":       "B (silent)",  "ba":      "BA",
    "be_bi":   "BE/BI",   "bo_bu":   "BO/BU",       "d":       "D (silent)",
    "da":      "DA",      "de_di":   "DE/DI",       "do_du":   "DO/DU",
    "e_i":     "E/I",     "g":       "G (silent)",  "ga":      "GA",
    "ge_gi":   "GE/GI",   "go_gu":   "GO/GU",       "h":       "H (silent)",
    "ha":      "HA",      "he_hi":   "HE/HI",       "ho_hu":   "HO/HU",
    "k":       "K (silent)", "ka":   "KA",           "ke_ki":   "KE/KI",
    "ko_ku":   "KO/KU",   "l":       "L (silent)",  "la":      "LA",
    "le_li":   "LE/LI",   "lo_lu":   "LO/LU",       "m":       "M (silent)",
    "ma":      "MA",      "me_mi":   "ME/MI",        "mo_mu":   "MO/MU",
    "n":       "N (silent)", "na":   "NA",           "ne_ni":   "NE/NI",
    "ng":      "NG (silent)", "nga": "NGA",          "nge_ngi": "NGE/NGI",
    "ngo_ngu": "NGO/NGU", "no_nu":   "NO/NU",        "o_u":     "O/U",
    "p":       "P (silent)", "pa":   "PA",           "pe_pi":   "PE/PI",
    "po_pu":   "PO/PU",   "s":       "S (silent)",  "sa":      "SA",
    "se_si":   "SE/SI",   "so_su":   "SO/SU",        "t":       "T (silent)",
    "ta":      "TA",      "te_ti":   "TE/TI",        "to_tu":   "TO/TU",
    "w":       "W (silent)", "wa":   "WA",           "we_wi":   "WE/WI",
    "wo_wu":   "WO/WU",   "y":       "Y (silent)",  "ya":      "YA",
    "ye_yi":   "YE/YI",   "yo_yu":   "YO/YU",
}

# ---------------------------------------------------------------------------
# Segmentation hyper-parameters
# ---------------------------------------------------------------------------
# Adaptive threshold block size (must be odd)
THRESH_BLOCK_SIZE = 25
THRESH_C          = 10       # Constant subtracted from mean

# Contour filtering — tune these if detection is too aggressive/permissive
MIN_CONTOUR_AREA     = 400   # px²  — ignore tiny specks
MAX_CONTOUR_AREA_PCT = 0.70  # ignore box that covers >70 % of image area
MIN_ASPECT_RATIO     = 0.15  # width / height
MAX_ASPECT_RATIO     = 6.5

# Bounding-box padding (px) added around each detected contour
BBOX_PAD = 6

# Annotation box colours cycling list  (BGR for OpenCV)
BOX_COLORS_BGR = [
    (  0, 180, 255),   # amber-ish
    ( 60, 210, 100),   # green
    (255, 140,  60),   # blue
    (200,  80, 220),   # purple
    ( 50, 200, 200),   # teal
    ( 80, 130, 255),   # salmon
]

# ---------------------------------------------------------------------------
# Model path — adjust if your .pth lives elsewhere
# ---------------------------------------------------------------------------
# Searches: Backend/models/, project root, then fails gracefully
_search_paths = [
    Path(__file__).parent / "models" / "rexnet_baybayin_final.pth",
    Path(__file__).parent.parent / "rexnet_baybayin_final.pth",
]
MODEL_PATH = next((p for p in _search_paths if p.exists()), None)

# ---------------------------------------------------------------------------
# Inference pre-processing transform (identical to val_transform in notebook)
# ---------------------------------------------------------------------------
_infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: torch.nn.Module | None = None
idx_to_class: dict[int, str] = {i: c for i, c in enumerate(BAYBAYIN_CLASSES)}
NUM_CLASSES: int = len(BAYBAYIN_CLASSES)


def load_model() -> None:
    """Load ReXNet-150 weights at startup. Runs once."""
    global model, idx_to_class, NUM_CLASSES

    log.info(f"Device: {device}")

    if MODEL_PATH is None:
        log.warning(
            "⚠️  rexnet_baybayin_final.pth not found. "
            "Running in DEMO mode — random predictions will be returned. "
            "Place the model file in Backend/models/ to enable real inference."
        )
        # Build architecture-only model so the app still starts
        model = timm.create_model("rexnet_150", pretrained=False, num_classes=NUM_CLASSES)
        model.to(device).eval()
        return

    log.info(f"Loading model from {MODEL_PATH} …")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Handle both raw state-dict and our training checkpoint dict
    if "model_state_dict" in checkpoint:
        state_dict   = checkpoint["model_state_dict"]
        NUM_CLASSES  = checkpoint.get("num_classes", NUM_CLASSES)
        idx_to_class = checkpoint.get("idx_to_class", idx_to_class)
        # Rebuild mapping if stored as str-keyed dict (JSON serialised)
        if idx_to_class and isinstance(next(iter(idx_to_class)), str):
            idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    else:
        state_dict = checkpoint

    model = timm.create_model("rexnet_150", pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    log.info(f"✅ ReXNet-150 loaded — {NUM_CLASSES} classes.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SulatAI — Baybayin Character Detection API",
    description=(
        "ReXNet-150–based detection and classification of handwritten "
        "Baybayin characters across all 59 modern syllabary classes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------
class Detection(BaseModel):
    id: int
    label: str                  # model class key e.g. "ba"
    transliteration: str        # human-readable e.g. "BA"
    bbox: List[int]             # [x_min, y_min, x_max, y_max]
    confidence: float           # 0.0 – 1.0


class PredictResponse(BaseModel):
    detections: List[Detection]
    annotated_image: str        # base64-encoded PNG
    processing_time_ms: float
    characters_found: int


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes → OpenCV BGR array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")
    return img


def _bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR → PIL RGB."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _encode_bgr_to_b64(bgr: np.ndarray) -> str:
    """Encode an OpenCV BGR image to a base64 PNG data-URI."""
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode annotated image.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Step B — Segmentation
# ---------------------------------------------------------------------------

def segment_characters(bgr: np.ndarray) -> List[dict]:
    """
    Detect individual Baybayin character bounding boxes using:
      1. Grayscale conversion
      2. Adaptive thresholding (Gaussian, THRESH_BINARY_INV)
      3. Morphological close (small gaps between strokes)
      4. Contour detection (RETR_EXTERNAL)
      5. Filter by area + aspect ratio

    Returns a list of dicts: {x, y, w, h}  sorted left-to-right, top-to-bottom.
    """
    h_img, w_img = bgr.shape[:2]
    total_area    = h_img * w_img

    # 1. Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 2. Adaptive threshold — handles uneven lighting across the page
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=THRESH_BLOCK_SIZE,
        C=THRESH_C,
    )

    # 3. Morphological close — bridges small gaps inside a single character
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: List[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        if area > total_area * MAX_CONTOUR_AREA_PCT:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if not (MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO):
            continue

        # Apply padding (clipped to image bounds)
        x1 = max(0, x - BBOX_PAD)
        y1 = max(0, y - BBOX_PAD)
        x2 = min(w_img, x + w + BBOX_PAD)
        y2 = min(h_img, y + h + BBOX_PAD)

        rois.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})

    # Sort: top-to-bottom, then left-to-right (reading order)
    rois.sort(key=lambda r: (r["y"] // 60, r["x"]))
    return rois


# ---------------------------------------------------------------------------
# Step C + D — Crop, preprocess, infer
# ---------------------------------------------------------------------------

@torch.inference_mode()
def classify_rois(bgr: np.ndarray, rois: List[dict]) -> List[dict]:
    """
    Crop each ROI, run through ReXNet-150, return prediction metadata.
    Batched inference for efficiency.
    """
    if not rois or model is None:
        return []

    tensors = []
    for roi in rois:
        crop_bgr = bgr[roi["y"]: roi["y"] + roi["h"], roi["x"]: roi["x"] + roi["w"]]
        pil_img  = _bgr_to_pil_rgb(crop_bgr)
        tensors.append(_infer_transform(pil_img))

    batch = torch.stack(tensors).to(device)       # (N, 3, 224, 224)
    logits = model(batch)                          # (N, 59)
    probs  = torch.softmax(logits, dim=1)          # (N, 59)
    top_probs, top_idx = probs.max(dim=1)          # (N,)

    results = []
    for i, roi in enumerate(rois):
        cls_idx    = top_idx[i].item()
        confidence = top_probs[i].item()
        label      = idx_to_class.get(cls_idx, BAYBAYIN_CLASSES[cls_idx % NUM_CLASSES])
        results.append({
            "roi":        roi,
            "label":      label,
            "confidence": confidence,
            "cls_idx":    cls_idx,
        })

    return results


# ---------------------------------------------------------------------------
# Step E — Annotate image
# ---------------------------------------------------------------------------

def annotate_image(bgr: np.ndarray, predictions: List[dict]) -> np.ndarray:
    """Draw bounding boxes + labels onto a copy of the image."""
    annotated = bgr.copy()
    for i, pred in enumerate(predictions):
        roi   = pred["roi"]
        color = BOX_COLORS_BGR[i % len(BOX_COLORS_BGR)]
        label = pred["label"].upper()
        conf  = int(pred["confidence"] * 100)
        text  = f"{label} {conf}%"

        x1, y1 = roi["x"], roi["y"]
        x2, y2 = roi["x"] + roi["w"], roi["y"] + roi["h"]

        # Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        label_y = y1 - 6 if y1 - 6 > th else y1 + th + 6
        cv2.rectangle(annotated, (x1, label_y - th - 2), (x1 + tw + 6, label_y + baseline), color, -1)

        # Label text (dark on coloured background)
        cv2.putText(
            annotated, text,
            (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (20, 20, 20),   # near-black
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "SulatAI Baybayin Detection API",
        "status":  "running",
        "model":   "ReXNet-150",
        "classes": NUM_CLASSES,
        "device":  str(device),
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(image: UploadFile = File(...)):
    """
    Main detection endpoint.

    - Accepts: PNG / JPEG / WEBP image upload (≤ 10 MB)
    - Returns: JSON with per-character detections + base64 annotated image
    """
    # ── Validate upload ──────────────────────────────────────────────────
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{image.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    raw = await image.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit.")

    t_start = time.perf_counter()

    # ── Decode ───────────────────────────────────────────────────────────
    try:
        bgr = _bytes_to_bgr(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # ── Step B : Segment ─────────────────────────────────────────────────
    rois = segment_characters(bgr)
    log.info(f"Segments found: {len(rois)}")

    # ── Step C + D : Classify ────────────────────────────────────────────
    predictions = classify_rois(bgr, rois)

    # ── Step E : Annotate ────────────────────────────────────────────────
    annotated_bgr = annotate_image(bgr, predictions)
    annotated_b64 = _encode_bgr_to_b64(annotated_bgr)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # ── Build response detections ─────────────────────────────────────────
    detections: List[Detection] = []
    for idx, pred in enumerate(predictions, start=1):
        roi   = pred["roi"]
        label = pred["label"]
        detections.append(
            Detection(
                id              = idx,
                label           = label,
                transliteration = CLASS_DISPLAY.get(label, label.upper()),
                bbox            = [roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]],
                confidence      = round(pred["confidence"], 4),
            )
        )

    log.info(f"Processed in {elapsed_ms:.1f} ms — {len(detections)} characters detected.")

    return PredictResponse(
        detections        = detections,
        annotated_image   = annotated_b64,
        processing_time_ms= round(elapsed_ms, 1),
        characters_found  = len(detections),
    )
