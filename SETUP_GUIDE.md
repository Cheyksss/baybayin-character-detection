# SulatAI — Complete Setup Guide for Groupmates

This guide walks through every step to get the system running on your local machine.

---

## Prerequisites

- **Windows 10/11** with PowerShell
- **Git** installed ([download](https://git-scm.com/download/win))
- **Python 3.9+** ([download from Microsoft Store](https://www.microsoft.com/store/productId/9NRWMJP3717K) or [python.org](https://www.python.org/downloads/))

---

## Step 1: Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/YOUR_REPO_URL/baybayin-character-detection.git
cd baybayin-character-detection
```

Replace `YOUR_REPO_URL` with your actual GitHub repository URL.

---

## Step 2: Extract the Trained Model (Critical!)

The system **requires** the trained ReXNet-150 model file: `rexnet_baybayin_final.pth` (~3.6 MB)

### Option A: Get from Groupmate (Recommended)

Ask the person who ran the training notebook to send you `rexnet_baybayin_final.pth`. They can find it in:
- **Google Drive** → `My Drive/baybayin_checkpoints/rexnet_baybayin_final.pth`

Once you receive it:

1. Place the file here:
   ```
   Backend/models/rexnet_baybayin_final.pth
   ```

2. Verify it's in the right place:
   ```powershell
   dir Backend/models/
   ```
   
   You should see:
   ```
   rexnet_baybayin_final.pth
   README.md
   ```

### Option B: Export from Training Notebook (If You Have Access)

If you have access to the training Colab notebook:

1. Open `Baybayin_Model.ipynb` in [Google Colab](https://colab.research.google.com)
2. **Skip to the final cells** (don't re-train)
3. Run the cell that loads the checkpoint:
   ```python
   CHECKPOINT_DIR = '/content/drive/MyDrive/baybayin_checkpoints/'
   MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_phase2_model.pth')
   ```
4. Run the final export cell (saves `rexnet_baybayin_final.pth`)
5. Download from Google Drive and place in `Backend/models/`

---

## Step 3: Install Python Dependencies

### 3a. Check Python is Installed

```powershell
python --version
```

You should see `Python 3.9.x` or higher.

### 3b. Navigate to Backend Directory

```powershell
cd Backend
```

### 3c. Install Requirements

```powershell
python -m pip install -r requirements.txt
```

This installs:
- `fastapi` — web framework
- `uvicorn` — server
- `torch` & `torchvision` — deep learning
- `opencv-python-headless` — image processing
- `timm` — model zoo
- `Pillow` — image handling

**Wait for this to complete.** On first install, it may take 5–15 minutes.

### 3d. Verify Installation

```powershell
python -m pip list
```

Look for these packages:
- `fastapi`
- `uvicorn`
- `torch`
- `opencv-python-headless`
- `timm`

---

## Step 4: Start the Backend Server

```powershell
python -m uvicorn app:app --reload
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Verify the Backend is Running

Open your browser and go to:
```
http://127.0.0.1:8000/docs
```

You should see an **interactive API explorer** with the `/predict` endpoint listed.

**Keep this PowerShell window open** while testing.

---

## Step 5: Open the Frontend

In a **new browser tab**, open:

```
file:///C:/Projects/baybayin-character-detection/Frontend/index.html
```

Or simply drag the `Frontend/index.html` file into your browser.

You should see the **SulatAI dashboard** with:
- Hero section
- Upload area
- Reference guide
- About section

---

## Step 6: Test the System

### 6a. Upload an Image

1. Click **"Try the Detector"** or drag an image into the upload box
2. Supported formats: PNG, JPEG, WEBP (≤ 10 MB)
3. Click **"Process Image"**

### 6b. View Results

The dashboard will show:
- **Original image** (left panel)
- **Annotated image** (right panel) with bounding boxes around detected characters
- **Statistics** (total characters, average confidence, unique classes, processing time)
- **Detection table** with character labels, transliterations, bounding boxes, and confidence scores

### 6c. Export Results

Click **"Export CSV"** to download a spreadsheet of all detections.

---

## Troubleshooting

### ❌ "rexnet_baybayin_final.pth not found"

**Error in backend logs:**
```
⚠️  rexnet_baybayin_final.pth not found. 
Running in DEMO mode — random predictions will be returned.
```

**Solution:** Get the model file from your groupmate and place it in `Backend/models/`.

### ❌ "Python not found"

**Error:**
```
python : The term 'python' is not recognized...
```

**Solution:** 
- Python is not installed or not in PATH
- Download from [python.org](https://www.python.org/downloads/)
- **During installation, check "Add Python to PATH"**
- Restart PowerShell after installing

### ❌ "ModuleNotFoundError: No module named 'torch'"

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
- Dependencies didn't install correctly
- Try again:
  ```powershell
  python -m pip install --upgrade -r requirements.txt
  ```

### ❌ "Port 8000 already in use"

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
- Another process is using port 8000
- Kill it:
  ```powershell
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```
- Or use a different port:
  ```powershell
  python -m uvicorn app:app --port 8001 --reload
  ```

### ❌ "CORS error in browser console"

**Error:**
```
Access to XMLHttpRequest blocked by CORS policy
```

**Solution:**
- Backend and frontend are on different origins
- Backend already has CORS enabled, but verify:
  1. Backend is running on `http://127.0.0.1:8000`
  2. Frontend is opening from `file://` protocol (local file)
  3. This is allowed by the CORS middleware in `app.py`

---

## Full System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER BROWSER                            │
│  Frontend/index.html + app.js + style.css (Tailwind)        │
│                                                              │
│  [Upload Image] → [Process] → [View Results & CSV Export]   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    HTTP POST /predict
                    (multipart/form-data)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FASTAPI BACKEND                             │
│              http://127.0.0.1:8000                           │
│                                                              │
│  Step B: OpenCV Segmentation                                │
│    → Grayscale → Adaptive Threshold → Contour Detection     │
│    → Filter by area & aspect ratio                          │
│                                                              │
│  Step C: ROI Processing                                      │
│    → Crop each character → Resize 224×224 → Normalize       │
│                                                              │
│  Step D: ReXNet-150 Inference                               │
│    → Load rexnet_baybayin_final.pth                         │
│    → Batch classify → Softmax confidence                    │
│                                                              │
│  Step E: Response                                            │
│    → JSON detections[] + base64 annotated image             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    HTTP 200 JSON
                           │
                    Render in frontend
```

---

## Running on Startup (Optional)

To avoid retyping commands every time, create a batch file:

**`start_backend.bat`** (save in project root):

```batch
@echo off
cd Backend
python -m uvicorn app:app --reload
pause
```

Then just double-click `start_backend.bat` to launch the server.

---

## Deployment (Future)

When ready for production:

1. **Backend** can be deployed to:
   - Heroku (free tier ended, use [Render](https://render.com) or [Railway](https://railway.app))
   - Google Cloud Run
   - AWS Lambda + API Gateway
   - Your own VPS

2. **Frontend** can be deployed to:
   - GitHub Pages
   - Netlify
   - Vercel
   - Any static hosting

3. **Model storage** (instead of local file):
   - HuggingFace Model Hub
   - Google Drive (via `gdown`)
   - AWS S3
   - Azure Blob Storage

---

## Questions?

- Check the main [README.md](README.md) for architecture overview
- Check [Backend/models/README.md](Backend/models/README.md) for model file details
- Check logs in browser console (F12) and terminal for errors

Happy detecting! 🚀
