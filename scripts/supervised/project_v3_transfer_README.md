# Project 3: Segmentation Transfer Learning (V2 -> V3)

This project contains the complete workflow for transferring the V2 segmentation model (trained on annotated data) to the new `标注2` dataset (fine-tuned via active learning).

## Directory Structure (Folder Layout)

We have organized the project into 4 clear sections:

### 1. Data (`01_Data`)
- **Repair_Workspace**: 
    - Contains the 5 "Golden Samples" that you manually corrected in LabelMe.
    - These corrections were the key to teaching the model the new logic.

### 2. Models (`02_Models`)
- **checkpoints**: Not just the old model, but the **fine-tuned** weights (`models_v3_finetuned.pth`) that incorporate your corrections.

### 3. Code (`03_Code`)
- Contains all Python scripts used for:
    - Prediction (`predict_full_images.py`)
    - Fine-tuning (`fine_tune_v3.py`)
    - Audit & Verification (`extract_v2_baseline.py`, `audit_model_change.py`)

### 4. Results (`04_Results`)
- **Final_Predictions**: 
    - The full set of 100 images processed by the evolved V3 model.
    - Contains `_pred.png` (Overlay) and `_mask.png` (Binary Mask) for every image.
- **Audit_Report**:
    - **comparison_report_v2_v3.md**: A detailed report analyzing the changes.
    - **change_audit.csv**: Statistical data for all 100 images.
    - **all_comparisons/**: 
        - **100 Side-by-Side Images**: `[ 原图 ] | [ 旧模型 V2 (红) ] | [ 新模型 V3 (绿) ]`
        - Use these to visually verify the model's performance.

---

## How to Verify
1. Go to `04_Results/Audit_Report/all_comparisons`.
2. Review the images.
    - **Green** represents the new model's prediction.
    - **Red** represents the old model's prediction.
3. If Green looks better than Red (fewer missing particles, better boundaries), the Transfer Learning was successful.

## How to Re-Run
If you need to run this on new data:
1. Copy new images to a folder.
2. Run `python 03_Code/predict_full_images.py --src_dir "YOUR_NEW_IMAGE_FOLDER"`
