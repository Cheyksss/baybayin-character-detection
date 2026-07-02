# SulatAI — Handwritten Baybayin Character Detection

> **Capstone Project 2026 · Group 2 · Adamson University**  
> A ReXNet-150–based system for detecting and classifying all 59 modern Baybayin syllabary characters from handwritten images.

---

## Project Structure

```
baybayin-character-detection/
├── Backend/
│   ├── app.py               ← FastAPI inference server (main backend)
│   ├── requirements.txt     ← Python dependencies
│   └── models/
│       ├── README.md        ← Where to place rexnet_baybayin_final.pth
│       └── rexnet_baybayin_final.pth   (you download this from Drive)
│
├── Frontend/
│   ├── index.html           ← Single-page dashboard
│   ├── app.js               ← Async API calls + UI logic
│   └── style.css            ← Custom styles (used alongside Tailwind CDN)
│
├── Baybayin_Model.ipynb     ← Training notebook (run on Google Colab)
├── baybayin_model.py        ← Auto-generated .py export of the notebook
└── README.md                ← This file
```

---

## Quick Start

### 1. Train the Model (Google Colab)

Open `Baybayin_Model.ipynb` in Google Colab and run all cells.  
The final cell saves `rexnet_baybayin_final.pth` to your Google Drive.  
Download it and place it in `Backend/models/`.

### 2. Start the Backend

```bash
cd Backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API will be available at `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

### 3. Open the Frontend

Open `Frontend/index.html` in your browser (no build step needed).  
The page connects to `http://127.0.0.1:8000/predict` automatically.

> **Offline / Demo mode:** If the backend is not running, the UI falls back to  
> a set of static example detections so the interface remains presentable.

---

## Detection Pipeline

| Step | Operation | Implementation |
|------|-----------|----------------|
| A | Receive multi-character image upload | FastAPI `UploadFile` |
| B | Grayscale → Adaptive Threshold → Contour detection → Filter by area & aspect ratio | OpenCV (`cv2`) |
| C | Crop ROIs, resize to 224×224, ImageNet normalise | Pillow + torchvision |
| D | Batch inference through ReXNet-150 | PyTorch + timm |
| E | Return JSON: bbox coords, transliteration, confidence + annotated image | FastAPI JSON response |

---

## API Reference

### `POST /predict`

**Request:** `multipart/form-data` with field `image` (PNG / JPEG / WEBP, ≤ 10 MB)

**Response:**
```json
{
  "detections": [
    {
      "id": 1,
      "label": "ba",
      "transliteration": "BA",
      "bbox": [42, 30, 140, 125],
      "confidence": 0.9712
    }
  ],
  "annotated_image": "data:image/png;base64,…",
  "processing_time_ms": 312.5,
  "characters_found": 6
}
```

### `GET /health`
Returns `{"status": "ok", "model_loaded": true}`

---

## Team

| Role | Name |
|------|------|
| CEO  | Jose Martin G. Cacao |
| CFO  | Jayq Andrei B. Gaguis |
| CTO  | David Jonathan T. Pomasin |
| COO/CMO | Mervyn Mario Simons |

**Adviser:** Paul Jacob C. Cruz — Data Science Capstone Professor
