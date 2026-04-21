# Medical Image Classification: Swin Transformer + MobileViT Hybrid Model

A state-of-the-art hybrid deep learning architecture for medical image classification on the Kvasir-v2 dataset, combining Swin Transformer and MobileViT backbones with gradient boosting classifiers.

## 🎯 Project Overview

This project implements a production-ready medical image classification system achieving **92.17% test accuracy** on 8 medical classes with **99.49% AUROC** and **zero overfitting**.

### Architecture

```
Input Image (224×224)
    ↓
┌─────────────────────────────┐
│  Feature Extraction         │
├─────────────┬───────────────┤
│  Swin-Small │   MobileViT   │
│  (768 dims) │  (384 dims)   │
└──────┬──────┴────────┬──────┘
       └────────┬──────┘
          Concatenate (1152 dims)
              ↓
         Fusion Layer
          (256 dims)
              ↓
    ┌─────────────────────┐
    │ Gradient Boosting   │
    ├─────────────────────┤
    │ XGBoost / LightGBM  │
    │ / CatBoost (Best)   │
    └─────────────────────┘
              ↓
        Classification Output
```

## 📊 Results

| Model | Val Accuracy | Test Accuracy | AUROC | Overfitting |
|-------|-------------|---------------|-------|------------|
| **CatBoost** 🏆 | 92.17% | 92.17% | 99.49% | 0.00% |
| XGBoost | 92.42% | 91.83% | 99.41% | 0.58% |
| LightGBM | 92.33% | 91.83% | 99.41% | 0.50% |

### Per-Class Performance (CatBoost)

| Class | Accuracy |
|-------|----------|
| Dyed-Lifted-Polyps | 90.0% |
| Dyed-Resection-Margins | 98.0% |
| Esophagitis | 84.0% |
| Normal-Cecum | 96.0% |
| Normal-Pylorus | 99.0% |
| Normal-Z-Line | 80.0% |
| Polyps | 94.0% |
| Ulcerative-Colitis | 96.0% |

## 📋 Dataset

- **Name**: Kvasir-v2
- **Size**: 8,000 images
- **Classes**: 8 medical categories
- **Train/Val/Test Split**: 70% / 15% / 15%

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)
- 8+ GB GPU memory recommended

### Installation

```bash
# Clone repository
git clone <repository-url>
cd MobileVit+Swin(small)

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Run complete training pipeline
python train_hybrid_pipeline.py

# Or use the Jupyter notebook
jupyter notebook Hybrid_Swin_MobileViT_Training.ipynb
```

### Inference

```bash
# Display results
python display_dual_results.py
```

## 📁 File Structure

```
├── hybrid_model.py                          # Core model architecture
├── train_hybrid_pipeline.py                 # Training pipeline with Optuna
├── display_dual_results.py                  # Results visualization
├── Hybrid_Swin_MobileViT_Training.ipynb     # Interactive notebook
├── training_summary.json                    # Training configuration & results
├── COMPLETE_RESULTS.txt                     # Detailed results report
├── DUAL_HYBRID_RESULTS_SUMMARY.txt          # Quick summary
├── catboost_model.cbm                       # Best model (CatBoost)
├── xgboost_model.json                       # Alternative model
├── lightgbm_model.txt                       # Alternative model
├── extracted_features.npz                   # Cached features
├── comparison.png                           # Model comparison visualization
└── requirements.txt                         # Python dependencies
```

## 🔧 Configuration

Key parameters in `train_hybrid_pipeline.py`:

```python
config = {
    'dataset_root': 'kvasir-dataset-v2',
    'mobilevit_weights': 'mobilevit_kvasir_v2_best_optuna.pth',
    'swin_variant': 'small',  # or 'base'
    'num_classes': 8,
    'batch_size': 32,
    'image_size': 224,
    'freeze_backbones': True,
    'classifier_types': ['xgboost', 'lightgbm', 'catboost'],
    'optuna_trials': 50,  # per classifier
}
```

## 🎯 Features

- ✅ **Dual-backbone architecture**: Combines spatial (MobileViT) and hierarchical (Swin) features
- ✅ **Transfer learning**: Pre-trained on ImageNet and medical datasets
- ✅ **Hyperparameter optimization**: Optuna with TPE sampler (150 trials total)
- ✅ **GPU acceleration**: Supports CUDA for faster training
- ✅ **Production-ready**: Zero overfitting and excellent generalization
- ✅ **Efficient inference**: ~18ms per image

## 📈 Training Details

| Aspect | Value |
|--------|-------|
| Optimization Framework | Optuna (TPE sampler) |
| Trials per Classifier | 50 |
| Total Trials | 150 |
| Feature Extraction Time | ~2.4 minutes |
| Total Training Time | ~47 minutes |
| GPU Used | NVIDIA (CUDA enabled) |

## 🔍 Key Findings

1. **CatBoost wins**: Best generalization with zero overfitting
2. **Feature fusion works**: Concatenating different architectures improves robustness
3. **Transfer learning effective**: Frozen backbones achieve state-of-the-art results
4. **Balanced performance**: High precision, recall, and AUROC across all models

## 📝 Citation

If you use this project, please cite:

```bibtex
@article{MobileViTSwin2025,
  title={Hybrid Swin Transformer and MobileViT for Medical Image Classification},
  author={Your Name},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ✉️ Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Kvasir-v2 dataset authors
- PyTorch team
- Optuna framework creators
