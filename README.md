# ![Deepfake Detection](https://img.shields.io/badge/Hybrid%20Deepfake%20Detection-CNN%2B3D%20Geometry-blue)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This repository implements a **multi-modal deepfake detection system** combining:

- **2D CNN (EfficientNet-B0)** – frame-level visual analysis  
- **3D Facial Geometry Analysis** – measurements from DECA meshes  
- **Biological Plausibility Scoring** – validates geometry against simulated population priors  
- **Score Fusion** – integrates CNN and geometric cues for final predictions  

**Problem:** Traditional CNN-based methods often miss subtle deepfakes due to lack of 3D structural verification. This project addresses this limitation by incorporating **geometric and biological plausibility scores**.

---

## Directory Structure

      deepfake-detection/
      ├── data/
      │ ├── raw/ # DFDC, FaceForensics++, CelebA datasets
      │ ├── merged/ # Merged dataset
      │ └── processed/ # Cropped face images
      ├── models/ # Trained CNN models
      ├── outputs/
      │ ├── meshes/ # DECA mesh outputs
      │ └── geometry_vectors/ # Geometry features extracted from meshes
      ├── scripts/
      │ ├── prepare_dataset.py
      │ ├── extract_faces.py
      │ ├── train_cnn.py
      │ ├── evaluate_cnn.py
      │ ├── extract_geometry.py
      │ ├── compute_plausibility.py
      │ ├── fuse_scores.py
      │ └── evaluate_pipeline.py
      ├── results/
      │ └── metadata.csv
      ├── README.md
      └── requirements.txt



---

## Installation

1. Clone the repository:

git clone https://github.com/monishasrinivas-04/deepfake-detection.git
cd deepfake-detection

Install dependencies:
pip install -r requirements.txt

**Dependencies:** Python 3.9+, PyTorch, torchvision, OpenCV, Trimesh, NumPy, Pandas, Matplotlib, MTCNN

---
## Dataset Preparation
    Download manually:
    DFDC: Kaggle
    FaceForensics++: GitHub
    CelebA: CUHK Multimedia Lab
## Organize and merge datasets:
    python scripts/prepare_dataset.py
    Creates real/ and fake/ directories
    Merges DFDC, FaceForensics++, and CelebA datasets

    Generates metadata.csv

## Extract face crops:

    python scripts/extract_faces.py
    Crops faces from sampled frames every 10 frames
    
    Saves images in data/processed/faces/real and data/processed/faces/fake

## CNN Training

          python scripts/train_cnn.py
          Model: EfficientNet-B0
          Data augmentation: random crop, horizontal flip, rotation, color jitter
          Optimizer: Adam, lr=1e-4
          Epochs: 15
          Batch size: 32
    Output: models/baseline_efficientnet.pth
---

## CNN Evaluation

    python scripts/evaluate_cnn.py
    Evaluates model accuracy on processed faces

  3D Geometry Extraction

    python scripts/extract_geometry.py
    DECA meshes from outputs/meshes/

    Computes facial measurements: jaw width, nose width, face width, ratios

    Saves .npy vectors in outputs/geometry_vectors/

Biological Plausibility Scoring

    python scripts/compute_plausibility.py --vertices_path <path_to_npy>
    Computes plausibility score (0–1) for each face geometry

Score Fusion & Final Prediction

      python scripts/fuse_scores.py
      Normalizes CNN and geometry scores
      
      Weighted fusion: fused = w_cnn * cnn_score + w_bio * (1 - bio_score)
      
      Applies threshold to classify video as real or fake

Evaluation

    
    python scripts/evaluate_pipeline.py
    Computes accuracy, ROC-AUC, confusion matrix
    Visualizes ROC curves and confusion matrices
    
    Evaluates fusion pipeline performance

---
##**Quick Commands**

   # 1. Prepare dataset
    python scripts/prepare_dataset.py
    
  # 2. Extract faces
    python scripts/extract_faces.py
    
  # 3. Train CNN
    python scripts/train_cnn.py
    
  # 4. Evaluate CNN
    python scripts/evaluate_cnn.py
    
  # 5. Extract geometry vectors from DECA meshes
    python scripts/extract_geometry.py
    
 # 6. Compute biological plausibility (example)
    python scripts/compute_plausibility.py --vertices_path outputs/geometry_vectors/example.npy
    
  # 7. Fuse scores & classify
    python scripts/fuse_scores.py
    
 # 8. Evaluate fusion pipeline
    python scripts/evaluate_pipeline.py

---
---
