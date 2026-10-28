# Carya AI 🫁

> AI-assisted chest X-ray analysis, explainable predictions, and clinician-ready report generation.

Carya is a medical AI application that analyzes chest X-ray images for **14 thoracic conditions**, visualizes model attention with **Grad-CAM**, and generates downloadable PDF reports. It includes a Streamlit interface for quick local use and a unified Flask application with a React frontend, REST API, and Doctor Connect video-call experience.

> **Medical disclaimer:** Carya is intended for research, education, and clinical decision-support prototyping only. It is **not a diagnostic device** and must not replace evaluation by a qualified healthcare professional.

## Demo

- **Web application:** [carya-iota.vercel.app](https://carya-iota.vercel.app)
- **Repository:** [github.com/Shivam200427/Carya](https://github.com/Shivam200427/Carya)

## Features

### AI-powered X-ray analysis

- Upload PNG, JPG, JPEG, BMP, or TIFF chest X-ray images
- Predict probabilities for 14 thoracic conditions
- Adjustable detection threshold
- Automatic image enhancement using percentile windowing, CLAHE, gamma correction, and normalization
- CUDA acceleration when available, with CPU fallback

### Explainable results

- Grad-CAM heatmaps showing regions that influenced the prediction
- Activation-based abnormal-region visualization
- Probability scores for every supported condition

### Report generation

- Radiology-style text summary
- Detected disease keywords
- PDF export with prediction results and visualizations
- Timestamped report files for later download

### Multiple application interfaces

- **Streamlit app:** simple all-in-one local interface
- **Unified Flask app:** serves the frontend, API, report downloads, and Doctor Connect pages
- **React frontend:** modern upload and results experience
- **Doctor Connect:** WebRTC video-call signaling and report sharing through Socket.IO
- **Command-line inference:** generate reports without starting a web server

## Supported Conditions

The model predicts the following 14 conditions:

1. Atelectasis
2. Cardiomegaly
3. Effusion
4. Infiltration
5. Mass
6. Nodule
7. Pneumonia
8. Pneumothorax
9. Consolidation
10. Edema
11. Emphysema
12. Fibrosis
13. Pleural Thickening
14. Hernia

## Technology Stack

### Machine learning

- Python
- PyTorch and TorchVision
- TorchXRayVision DenseNet-121 backbone
- SemaCheXFormer architecture
- Transformer attention blocks
- Convolutional squeeze-and-excitation blocks
- Grad-CAM explainability

### Backend and interfaces

- Flask
- Flask-SocketIO
- Streamlit
- React 18
- Vite
- Material UI
- WebRTC video communication

### Image and report processing

- OpenCV
- Albumentations
- NumPy
- Pillow
- Matplotlib
- ReportLab
- Einops

## Repository Structure

```text
Carya/
├── Model/                         # Trained model checkpoint(s)
├── frontend/                      # React/Vite frontend
│   ├── src/                        # Frontend source code
│   ├── dist/                       # Production build output
│   └── package.json                # Frontend dependencies and scripts
├── templates/                     # Flask HTML templates
├── static/                        # Static assets and video-call files
├── video-call-app/                # Video-call reference application
├── reports/                       # Generated PDF reports
├── video_uploads/                 # PDFs shared during consultations
├── app.py                         # Unified Flask + Socket.IO application
├── flask_backend.py               # Standalone Flask inference backend
├── streamlit_app.py               # Streamlit application
├── chest_xray_inference.py        # Model, inference, Grad-CAM, and PDF reports
├── preprocess_xray_gui.py         # X-ray enhancement and preprocessing utilities
├── requirements.txt               # Python dependencies
├── README_INFERENCE.md            # CLI inference documentation
├── README_STREAMLIT.md             # Streamlit documentation
└── README_UNIFIED.md               # Unified Flask/React application documentation
```

## Prerequisites

- Python **3.10+** recommended
- Node.js **18+** for building the React frontend
- A trained model at `Model/final_model.pth`
- Optional NVIDIA GPU with a compatible PyTorch/CUDA installation
- `tkinter` only if using the command-line file picker

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Shivam200427/Carya.git
cd Carya
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

For GPU inference, install the PyTorch build appropriate for your CUDA version before installing or upgrading the remaining dependencies.

### 4. Install frontend dependencies

The React frontend has its own dependencies:

```bash
cd frontend
npm install
cd ..
```

> The repository's Python requirements cover the ML and Streamlit stack. The unified Flask application additionally imports Flask, Flask-SocketIO, and an asynchronous Socket.IO worker such as Eventlet; install the versions appropriate for your deployment if they are not already present in your environment.

## Running the Application

### Option A: Streamlit interface

The Streamlit app provides image upload, analysis, report history, Doctor Connect UI, and report downloads.

```bash
streamlit run streamlit_app.py
```

Open the displayed local URL, typically `http://localhost:8501`.

### Option B: Unified Flask + React application

Build the frontend first:

```bash
cd frontend
npm install
npm run build
cd ..
```

Start the unified server:

```bash
python app.py
```

Available pages and endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | React frontend or Flask fallback page |
| `GET` | `/doctor` | Doctor Connect page |
| `GET` | `/video-call` | Direct video-call interface |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Upload an X-ray and download a PDF report |
| `POST` | `/api/predict_json` | Upload an X-ray and receive JSON results |
| `GET` | `/api/download_report/<filename>` | Download a generated report |
| `POST` | `/api/upload` | Upload a PDF for consultation sharing |
| `GET` | `/api/video_uploads/<filename>` | Serve a shared PDF |

The server runs on `http://localhost:5000` by default.

### Option C: Command-line inference

Run inference on a specific image and generate a PDF report:

```bash
python chest_xray_inference.py \
  --image path/to/chest_xray.png \
  --model Model/final_model.pth \
  --output Chest_Report.pdf \
  --threshold 0.5
```

Available options:

```text
--image       Path to the X-ray image. If omitted, a file picker opens.
--model       Model checkpoint path. Default: Model/final_model.pth
--output      Output PDF path.
--threshold   Detection threshold from 0.0 to 1.0. Default: 0.5
--device      cuda or cpu. Auto-detected when omitted.
```

## Model and Inference Pipeline

Carya uses the `SemaCheXFormer` architecture:

```text
Chest X-ray
    ↓
Image preprocessing and enhancement
    ↓
TorchXRayVision DenseNet-121 feature extractor
    ↓
Transformer attention with label embeddings
    ↓
Convolutional squeeze-and-excitation refinement
    ↓
Multi-label classification head
    ↓
Disease probabilities
    ↓
Grad-CAM visualization + PDF report
```

The inference pipeline expects a model checkpoint compatible with the architecture defined in `chest_xray_inference.py`. The default path is:

```text
Model/final_model.pth
```

## API Example

Generate a PDF report with `curl`:

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "file=@path/to/chest_xray.png" \
  -F "threshold=0.5" \
  --output Chest_Report.pdf
```

Request JSON results instead:

```bash
curl -X POST http://localhost:5000/api/predict_json \
  -F "file=@path/to/chest_xray.png" \
  -F "threshold=0.5"
```

A successful JSON response includes:

- `predictions`
- `probabilities`
- `detected_diseases`
- `keywords`
- `report_text`
- Generated report metadata

## Frontend Development

Run the Vite development server:

```bash
cd frontend
npm run dev
```

Build the production frontend:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

When deployed through Flask, the built files in `frontend/dist/` are served by `app.py`.

## Configuration

Important defaults are defined in the application files:

- Model path: `Model/final_model.pth`
- Detection threshold: `0.5`
- Flask port: `5000`
- Maximum uploaded image size: `16 MB`
- Generated reports: `reports/`
- Video-call uploads: `video_uploads/`

For production deployments, set a secure Flask secret:

```bash
# Linux/macOS
export SECRET_KEY="replace-with-a-long-random-secret"

# Windows PowerShell
$env:SECRET_KEY="replace-with-a-long-random-secret"
```

Do not commit private keys, credentials, patient data, or sensitive medical reports to the repository.

## Troubleshooting

### Model file not found

Verify that the checkpoint exists at:

```text
Model/final_model.pth
```

You can also provide a different path using the `--model` command-line option or the application settings.

### Frontend does not load

Build the React application before starting Flask:

```bash
cd frontend
npm install
npm run build
```

Confirm that `frontend/dist/index.html` exists.

### Import errors

Reinstall the Python dependencies:

```bash
pip install -r requirements.txt
```

For the unified Flask application, also verify that Flask, Flask-SocketIO, and Eventlet are installed.

### CUDA or memory errors

- Run on CPU with `--device cpu`
- Reduce concurrent requests
- Install a PyTorch build compatible with the installed CUDA version
- Confirm that the model fits in available GPU memory

### Video calls do not connect

- Allow camera and microphone access in the browser
- Ensure both participants use the same room ID
- Check firewall and NAT settings
- Configure STUN/TURN infrastructure for production deployments
- Use an appropriate Socket.IO worker instead of Flask's development server

## Development Notes

- Generated reports and uploads are stored locally by default.
- Session history in the Streamlit app is in-memory and resets when the session restarts.
- The default Flask application uses in-memory room state for report sharing.
- The included development server is not intended for production use.
- Validate all model outputs with qualified medical professionals before any real-world use.

## Related Documentation

- [Command-line inference guide](README_INFERENCE.md)
- [Streamlit guide](README_STREAMLIT.md)
- [Unified application guide](README_UNIFIED.md)

## License

No license is currently specified for this repository. All rights are reserved unless the repository owner adds a license. Contact the repository owner before redistributing, modifying, or using this project commercially.
