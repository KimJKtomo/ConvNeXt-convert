# 🦴 Pediatric Fracture Classification with ConvNeXt

A PyTorch-based fracture classification pipeline using **ConvNeXt V2** as the backbone, optimized for **age group-specific training** on pediatric wrist X-ray datasets.

---

## 📌 Features

- ✅ **ConvNeXt V2 Backbone** (`timm`-based)
- ✅ Age group-wise training (`0–1.5`, `1.5–5`, `5–10`, `10–15`, `15–19`)
- ✅ Class imbalance handling via `WeightedRandomSampler` or `FocalLoss`
- ✅ Custom age group classifier using Swin Transformer
- ✅ Grad-CAM visualization per age group
- ✅ MLflow integration for logging metrics & saving best models

---

## 🗂️ Project Structure

```
.
├── train_fracture_per_agegroup_0703.py     # Age group-specific fracture training (ConvNeXt V2)
├── train_age_classifier_old.py             # Age group classifier (Swin-T)
├── Generate_Gradcam_Agegroup.py            # Grad-CAM visualization script
├── run_all_training_0704.py                # Full pipeline automation script
├── unified_dataset_0704.py                 # Dataset class with mask removal logic
├── age_split_testset_0704.py               # Fixed 1-year interval testset generator
├── age_train_tmp.csv / age_val_tmp.csv     # Train/val split CSVs
└── test_set_0704.csv                       # Final fixed test set (used in Grad-CAM eval)
```

---

## 🧪 Age Group Split

The fracture models are trained **separately per age group**, defined as:

| Age Group | Age Range (years)  |
|-----------|---------------------|
| 0         | 0 ~ 1.5             |
| 1         | 1.5 ~ 5             |
| 2         | 5 ~ 10              |
| 3         | 10 ~ 15             |
| 4         | 15 ~ 19             |

> These age boundaries are customizable in `AGE_GROUP_FN`.

---

## 🚀 How to Run

```bash
# Run full pipeline (split -> train classifier -> train per age group -> grad-cam)
python run_all_training_0704.py
```

Or run individual steps:

```bash
python age_split_testset_0704.py        # [Optional] 1y fixed test set
python generate_age_trainval_split.py   # Random train/val split
python train_age_classifier_old.py      # Train age classifier
python train_fracture_per_agegroup_0703.py  # Train per-age-group fracture models
python Generate_Gradcam_Agegroup.py     # Grad-CAM visualization
```

---

## 🧠 Model Backbone

This pipeline uses:
- `ConvNeXt V2` as the main image encoder for fracture classification
- `Swin-L` as the backbone for age group classification

All models are loaded from [timm](https://github.com/huggingface/pytorch-image-models).

---

## 📊 Example Output

Each age group training logs:
- Accuracy & F1 per epoch
- Confusion matrix per group
- Grad-CAM images saved under `gradcam_results/`

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

Required packages include:
- `torch`, `timm`, `opencv-python`, `pandas`, `scikit-learn`, `mlflow`, `pytorch-grad-cam`

---

## 👨‍⚕️ Dataset

The dataset is based on pediatric wrist X-rays (Kaggle/GRAZPEDWRI-DX compatible) with following fields:
- `image_path`, `age`, `fracture_visible`, `metal`, `ao_classification`, etc.

The dataset is cleaned to **exclude metal** cases and include **balanced fracture/normal** pairs per age.

---

## 🧼 Mask Removal (Optional)

Bounding boxes labeled with YOLO class `8` (mask region) are removed from input images during dataset loading.
