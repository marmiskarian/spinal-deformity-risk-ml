# spinal-deformity-risk-ml
Machine learning models for detecting and predicting spinal deformity risks using posture, biomechanical, and musculoskeletal data.

## Setup

```bash
conda create -n spinal-deformity-risk-ml python=3.11
conda activate spinal-deformity-risk-ml
pip install -r requirements.txt
```

## Data

**Current setup:** use the local file `msd_risk_dataset.xlsx` in the repo root (or in `data/`). 

- **Exploratory analysis:** `Scoliosis_progression_risk_detection.ipynb`
- **Train & evaluate predictor:** `train_and_evaluate.py`

## Run

**Train and test the risk prediction model:**
```bash
python train_and_evaluate.py
```
Ensure `msd_risk_dataset.xlsx` is in the repo root (or in `data/`). The script will:
- Load the data, split 80% train / 20% test (stratified)
- Preprocess: scale numeric features, one-hot encode Gender and Backpack_Position
- Train a Random Forest classifier
- Print test accuracy, classification report, confusion matrix, and top feature importances

Dataset (for reference): [mrhmnshu/musculoskeletal-disorders-risk-in-students](https://www.kaggle.com/datasets/mrhmnshu/musculoskeletal-disorders-risk-in-students). 
