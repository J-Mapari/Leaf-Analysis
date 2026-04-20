# Leaf Analysis Pipeline

## Overview
This project implements a full pipeline for:
- Leaf segmentation
- Feature extraction
- Machine learning classification

---

## Datasets Used

### 1. Flavia Dataset
- Clean, white background
- Used for baseline validation

### 2. PlantVillage Dataset
- Semi-natural images
- Used for robustness testing

### 3. ImageCLEF Plant Dataset
- Real-world images with metadata
- Used for advanced evaluation

### 4. UCI Leaf Dataset
- Feature-only dataset
- Used for feature validation

---

## Pipeline

1. Crop leaf region (remove background/hand)
2. Preprocess (CLAHE + blur)
3. Segment leaf (HSV + contour)
4. Extract features:
   - area
   - perimeter
   - aspect ratio
   - circularity
   - solidity
5. Train ML model (Random Forest)

---

## Machine Learning

Model: Random Forest

Input:
- Extracted features

Output:
- Predicted plant species

---

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py