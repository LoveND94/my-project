# CLMNet: Colon Cancer Liver Metastasis Prediction Model

This project provides a deep learning framework named **CLM-Net** for the automatic recognition and prognosis prediction of **colorectal liver metastasis (CRLM)** based on pathological images.

## 📦 Model Components
- `clmnet_model.py`: The CLM-Net model combining multi-scale feature extraction, attention mechanism, and CRF post-processing.
- `train_model.py`: Training script for the model.
- `evaluate.py`: Evaluation script using Accuracy, AUC, and F1 Score.
- `dataset.py`: Custom PyTorch dataset loader for pathological image data.

## 🧪 Features
- Multi-scale ASPP-based feature extraction
- Spatial Attention mechanism
- Conditional Random Field (CRF) post-processing
- Integrated classification & segmentation for CRLM
- Trained and validated using TCIA & Kaggle datasets

## 🛠 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python train_model.py        # Train the model
python evaluate.py           # Evaluate model performance
```

## 🧬 Dataset
The dataset consists of preprocessed pathological image slices of CRLM patients. You may place your `.png` or `.jpg` files in a folder like `train_data/`.

## 📈 Performance
CLM-Net achieved:
- Accuracy: **95%**
- AUC: **0.97**
- F1 Score: **94%**

## 🔬 Citation
If you use this model in academic work, please cite:

> "Deep learning for automatic pathology feature recognition and prognosis prediction in colorectal liver metastasis", MA-24-015, 2025.
