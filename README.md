---
title: Violence Detection App
emoji: "🎬"
colorFrom: "blue"
colorTo: "green"
sdk: "streamlit"
sdk_version: "1.26.0"   # adjust to your Streamlit version
python_version: "3.10"
app_file: "app/app.py"
pinned: false
---

# Violence Detection in Videos 

A deep learning project for **video violence detection** using a hybrid model combining **3D CNN (ResNet-18)** for spatial-temporal feature extraction and a **Transformer** for temporal modeling. Includes a **Streamlit app** for uploading videos and classifying them as **Violent** or **Non-Violent** in real-time.

---

## Features
- **Hybrid Video Classifier**: 3D CNN backbone + Transformer for temporal modeling.
- **Pretrained Model**: Ready-to-use `hybrid_model3.pth` for inference.
- **Robust Video Processing**: Handles variable-length videos, corrupted frames, grayscale/multi-channel input.
- **Streamlit App**: Simple interface for video classification.
- **Modular Structure**: Clear separation between model, utils, configs, and app.

---

##  Project Structure
Violence-Detection-video/
│
├── app/
│ └── app.py # Streamlit web app
│ └── __init__.py
├── data/
│ └── test videos # videos used for testing 
├── models/
│ ├── hybrid_model3.pth # Trained model checkpoint
│ └── model.py # Hybrid model definition
│ ├── vit_baseline.pth # Trained vit baseline model
│ └── vit_baseline_ResNet.pth # Trained ResNet model 
│ └── __init__.py
├── src/
│ └── video_utils.py # Video preprocessing utilities
│ └── load_model.py # Load trained model helper
│ └── __init__.py
├── configs/
│ └── config.py # Dataset, preprocessing, and training configs
│ └── __init__.py
├── notebooks/
│ ├── 01-1-data-preparation-and-vit-model-baseline(ResNet).ipynb
│ └── 01-2-data-preparation-and-vit-model-baseline(simple).ipynb
│ ├── 02-training-pipeline.ipynb
│ └── 02-training-pipeline-evaluation.ipynb
├── requirements.txt
└── README.md
└── .gitignore
└── LICENSE 

---

## Dataset & Preprocessing
- Merged from Kaggle: violent vs non-violent videos.
- Videos processed on-the-fly in the app: extract `NUM_FRAMES`, resize, normalize.
- **Cached `.pt` datasets are only needed for training**; app does not require caching.

---

## Model Architecture
- **Backbone**: `r3d_18` (3D ResNet-18), removes classifier, frozen except optional last block.
- **Temporal Transformer**: 3 layers, `hidden_dim=512`, `nhead=8`.
- **Classification Head**: Linear layer outputs single logit with `BCEWithLogitsLoss`.
- **Forward pass**: `[B, F, C, H, W] -> Backbone -> Transformer -> Classifier -> logit`

> Note: Current model repeats backbone features across frames; works fine for deployment.

---

## Installation and Usage 
```bash
git clone https://github.com/riham-esi/Violence-Detection-video.git
cd Violence-Detection
pip install -r requirements.txt
streamlit run app/app.py
 - Upload a video.
 - Get violence probability prediction.
```
## Training (Optional)
```bash
Use the notebooks in notebooks/:
01-1-data-preparation-and-vit-model-baseline(simple).ipynb → Prepare and cache videos and train simple model
01-2-data-preparation-and-vit-model-baseline(ResNet).ipynb → Prepare and cache videos and train ResNet model.
02-training-pipeline.ipynb → Train or fine-tune the model with pretrained hybrid model.
02-training-pipeline-evaluation.ipynb →Add evaluation to the final model trained 
```
---

## Evaluation Metrics
```bash
Accuracy
Precision
Recall
F1-score
Confusion_matrix ( TP, TN, FP, FN)
```
---
## Requirements
```bash
Python 3.8+
PyTorch
Torchvision
OpenCV
Streamlit
Numpy
```