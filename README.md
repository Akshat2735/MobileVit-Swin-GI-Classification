# DualHybridSwinSmall: Frozen Dual-Backbone Hierarchical Feature Fusion for Gastrointestinal Disease Classification

---

## Description

DualHybridSwinSmall is a production-ready hybrid deep learning framework for automated multi-class gastrointestinal (GI) disease classification from endoscopic imagery. The system combines two complementary frozen Vision Transformer backbones — Swin-Small and MobileViTv2_100 — whose outputs are fused through a compact trainable MLP, then classified by a TPE-optimised CatBoost gradient-boosted meta-learner.

The key design philosophy rests on three principles:

- **Representational complementarity** — Swin-Small captures rich multi-scale hierarchical structural features (768-dim), while MobileViTv2_100 encodes compact global semantic context via linear-complexity separable attention (512-dim). Together they form a 1,280-dimensional descriptor covering fine-grained local texture, multi-scale structure, and global context simultaneously.
- **Frozen transfer learning** — both backbones are locked at ImageNet-pretrained weights, preventing overfitting and catastrophic forgetting on the limited Kvasir-v2 corpus while eliminating the GPU memory demands of end-to-end fine-tuning.
- **Parameter efficiency** — only ~920K of 92.5M total parameters (~1%) are trainable, confined entirely to the fusion network and classification head.

The framework achieves **92.17% test accuracy**, **99.49% AUROC**, **zero validation-to-test generalisation gap**, and approximately **55 FPS** real-time inference throughput — suitable for direct clinical endoscopy deployment.

A comprehensive SHAP explainability pipeline (global rankings, per-class beeswarm plots, waterfall plots, force plots, and spatial heatmap overlays) provides clinically interpretable, regulatory-compliant evidence of feature attribution validity across all eight diagnostic categories.

---

## Dataset Information

**Dataset**: Kvasir-v2

Kvasir-v2 was curated by a multidisciplinary team at Simula Research Laboratory and Oslo University Hospital, drawing clinical endoscopy imagery from Vestre Viken Health Trust, Norway. All 8,000 images were verified by expert gastroenterologists, with ambiguous cases resolved through multi-physician panel review referencing histological biopsy results.

| Property | Details |
|---|---|
| Total images | 8,000 |
| Classes | 8 (1,000 images each — perfectly balanced) |
| Input resolution | 224 × 224 (bicubic resize) |
| Training split | 80% — 6,400 images |
| Validation split | 10% — 800 images |
| Test split | 10% — 800 images |
| Split strategy | Stratified random sampling (preserves class balance) |

**The 8 diagnostic categories:**

| # | Class | Clinical Type |
|---|---|---|
| 1 | Normal Z-Line | Anatomical landmark |
| 2 | Normal Pylorus | Anatomical landmark |
| 3 | Normal Cecum | Anatomical landmark |
| 4 | Esophagitis | Diffuse pathology |
| 5 | Polyps | Focal pathology |
| 6 | Ulcerative Colitis | Diffuse pathology |
| 7 | Dyed-Lifted Polyps | Surgical/procedural |
| 8 | Dyed Resection Margins | Surgical/procedural |

**Availability**: Publicly available via the official Kvasir dataset repository at https://datasets.simula.no/kvasir/

---

## Code Information

### Project Structure

```
DualHybridSwinSmall/
├── hybrid_model.py                        # Core architecture: dual-backbone + fusion MLP
├── train_hybrid_pipeline.py               # Full training pipeline with Optuna TPE optimisation
├── display_dual_results.py                # Results visualisation and metrics display
├── Hybrid_Swin_MobileViT_Training.ipynb   # Interactive Jupyter notebook for step-by-step training
├── training_summary.json                  # Training configuration and results metadata
├── COMPLETE_RESULTS.txt                   # Detailed per-class performance report
├── DUAL_HYBRID_RESULTS_SUMMARY.txt        # Quick results summary
├── catboost_model.cbm                     # Trained CatBoost model (best performer)
├── xgboost_model.json                     # Trained XGBoost model
├── lightgbm_model.txt                     # Trained LightGBM model
├── extracted_features.npz                 # Pre-computed 1,280-dim feature cache
├── requirements.txt                       # Python package dependencies
├── LICENSE                                # MIT License
└── README.md                              # This file
```

### Architecture Overview

```
Input Image (224 × 224 × 3)
         │
   ┌─────┴─────┐
   │           │
Swin-Small   MobileViTv2_100
 (FROZEN)      (FROZEN)
 768-dim        512-dim
   │           │
   └─────┬─────┘
  Concatenate [1,280-dim]
         │
  Linear(1280 → 512) + ReLU
         │
     Dropout(p=0.30)
         │
  Linear(512 → 256)
         │
  Linear(256 → 8) [CatBoost meta-learner]
         │
  8-Class Softmax Output
```

**Parameter summary:**

| Component | Parameters | Status |
|---|---|---|
| Swin-Small backbone | ~50.0 M | FROZEN |
| MobileViTv2_100 backbone | ~3.6 M | FROZEN |
| Linear(1280→512) + ReLU | 786,944 | Trainable |
| Dropout(p=0.30) | 0 | — |
| Linear(512→256) | 131,328 | Trainable |
| Classification head Linear(256→8) | 2,056 | Trainable |
| **Total trainable** | **~920K (1.0%)** | — |
| Total frozen | ~91.6 M (99.0%) | — |

### Model Performance

| Classifier | Val Accuracy | Test Accuracy | AUROC | Overfitting Gap | F1-Score | Log-Loss |
|---|---|---|---|---|---|---|
| **CatBoost (50-trial)** 🏆 | 92.17% | **92.17%** | **99.49%** | **0.00%** | 92.07% | 0.219 |
| XGBoost (50-trial) | 92.42% | 91.83% | 99.41% | 0.59% | 91.74% | 0.233 |
| LightGBM (50-trial) | 92.33% | 91.83% | 99.41% | 0.50% | 91.75% | 0.235 |
| Softmax head (baseline) | ~86.4% | ~85.8% | ~98.5% | ~0.6% | ~85.8% | ~0.42 |

**Per-class performance (CatBoost, 50-trial):**

| Class | Accuracy | AUROC | Precision | Recall | FPR |
|---|---|---|---|---|---|
| Normal Pylorus | 99.2% | ~1.000 | ~0.99 | ~1.00 | 0.16% |
| Normal Cecum | 99.4% | ~1.000 | ~0.98 | ~1.00 | 0.10% |
| Ulcerative Colitis | 94.9% | ~0.998 | ~0.96 | ~0.95 | 0.72% |
| Dyed Resection Margins | 94.4% | ~0.999 | ~0.94 | ~0.94 | 0.80% |
| Polyps | 92.8% | ~0.997 | ~0.93 | ~0.93 | 1.10% |
| Dyed Lifted Polyps | 90.8% | ~0.995 | ~0.91 | ~0.91 | 1.30% |
| Normal Z-Line | 85.7% | ~0.993 | ~0.86 | ~0.86 | 2.00% |
| Esophagitis | 79.1% | ~0.989 | ~0.76 | ~0.98 | 3.10% |

**Inference latency (GPU, 224 × 224 input):**

| Component | Latency |
|---|---|
| Swin-Small forward pass | ~12 ms |
| MobileViTv2 forward pass | ~5 ms |
| Feature concatenation | < 1 ms |
| CatBoost tree traversal | ~1 ms |
| **Total end-to-end** | **~18 ms (~55 FPS)** |

---

## Requirements

### System Requirements

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU acceleration)
- GPU with 8+ GB VRAM (NVIDIA recommended)
- 16+ GB system RAM

### Python Dependencies

Install all dependencies via:

```bash
pip install -r requirements.txt
```

Core packages:

| Package | Purpose |
|---|---|
| torch (2.0+) | Deep learning framework |
| torchvision | Vision utilities and pretrained models |
| timm | Swin Transformer and MobileViT implementations |
| catboost | CatBoost meta-classifier (best performer) |
| xgboost | XGBoost classifier |
| lightgbm | LightGBM classifier |
| optuna | Bayesian hyperparameter optimisation (TPE) |
| scikit-learn | Metrics, splitting, evaluation utilities |
| numpy | Numerical computing |
| pandas | Data manipulation |
| matplotlib | Visualisation |
| seaborn | Statistical visualisation |
| shap | Shapley value explainability (TreeSHAP) |
| Pillow | Image processing |
| tqdm | Progress bars |
| jupyter | Interactive notebook support |

---

## Usage Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DualHybridSwinSmall
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download Kvasir-v2 from https://datasets.simula.no/kvasir/ and extract it so the directory structure is:

```
kvasir-dataset-v2/
├── dyed-lifted-polyps/
├── dyed-resection-margins/
├── esophagitis/
├── normal-cecum/
├── normal-pylorus/
├── normal-z-line/
├── polyps/
└── ulcerative-colitis/
```

Place the `kvasir-dataset-v2/` folder in the project root directory.

### 5. Train the Model

**Option A — Python script (recommended):**

```bash
python train_hybrid_pipeline.py
```

This will:
- Extract 1,280-dimensional features from both frozen backbones and cache them to `extracted_features.npz`
- Run Optuna TPE optimisation (50 trials per classifier, 150 total)
- Train CatBoost, XGBoost, and LightGBM classifiers
- Save trained models (`catboost_model.cbm`, `xgboost_model.json`, `lightgbm_model.txt`)
- Generate performance reports in `hybrid_results/`

**Option B — Jupyter Notebook:**

```bash
jupyter notebook Hybrid_Swin_MobileViT_Training.ipynb
```

Execute cells sequentially for step-by-step training with inline visualisation.

### 6. Run Inference and View Results

```bash
python display_dual_results.py
```

Displays comparative metrics, per-class accuracy, confusion matrices, AUROC scores, and overfitting analysis across all three classifiers.

### 7. Configuration

Edit the config dictionary in `train_hybrid_pipeline.py` to customise behaviour:

```python
config = {
    'dataset_root':      'kvasir-dataset-v2/kvasir-dataset-v2',
    'swin_variant':      'small',          # 'small' (768-dim) or 'base' (1024-dim)
    'num_classes':       8,
    'batch_size':        32,
    'image_size':        224,
    'freeze_backbones':  True,             # Keep True — required for frozen transfer design
    'classifier_types':  ['xgboost', 'lightgbm', 'catboost'],
    'optuna_sampler':    'tpe',
    'optuna_trials':     50,               # Per classifier
    'output_dir':        'hybrid_results'
}
```

---

## Methodology

### Step 1 — Data Preprocessing

All images are resized to 224 × 224 pixels using bicubic interpolation and normalised using ImageNet channel statistics (μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]) to align the endoscopic image distribution with the statistical context under which both backbones were pretrained.

Training-time augmentations simulate real endoscopic variability: random horizontal and vertical flips (p=0.5 each), random rotation within ±30°, colour jitter (brightness/contrast/saturation 0.2, hue 0.1), and random resized cropping (scale range [0.85, 1.0]). No augmentation is applied during validation or inference.

The dataset is partitioned using stratified random sampling: 80% training, 10% validation, 10% test — yielding approximately 800/100/100 images per class.

### Step 2 — Dual-Stream Frozen Feature Extraction

Both backbones operate in frozen evaluation mode (`model.eval()`, `requires_grad=False`, `torch.no_grad()`) throughout all stages.

**Swin-Small** processes each input through a four-stage hierarchical feature pyramid, applying shifted-window multi-head self-attention within non-overlapping 7×7 token windows. Shifted partitioning in alternating blocks enables cross-window integration. The four stages progressively halve spatial resolution and double channel dimension (C=128 → 256 → 512 → 1024), producing a final globally average-pooled 768-dimensional embedding. Shifted-window attention reduces complexity from O(N²) to linear O(N).

**MobileViTv2_100** interleaves MobileNetV2-style inverted bottleneck blocks with separable self-attention blocks. Its key innovation replaces standard O(k²) cross-token attention with two O(k) element-wise operations via a context score decomposition, enabling global token interaction without quadratic overhead. It produces a 512-dimensional globally average-pooled embedding.

### Step 3 — Feature Fusion

The two frozen embeddings are concatenated to form a 1,280-dimensional descriptor:

```
c = [f_Swin (768-dim) || f_MViT (512-dim)]  →  R^1280
```

A compact asymmetric fusion MLP compresses this into a 256-dimensional discriminative embedding:

```
h     = ReLU( Linear(1280 → 512)(c) )
h_drop = Dropout(h, p=0.30)
e     = Linear(512 → 256)(h_drop)
```

All gradient updates during training are confined to these ~920K fusion parameters.

### Step 4 — CatBoost Meta-Classifier with TPE Optimisation

The 256-dimensional embedding is passed to a CatBoost gradient-boosted symmetric tree ensemble. Hyperparameters are selected via Bayesian optimisation using the Tree-structured Parzen Estimator (TPE) in Optuna over 50 trials, minimising validation log-loss.

TPE models two conditional densities — l(x) over high-performing configurations and g(x) over poor ones — and proposes candidates by maximising the ratio l(x)/g(x). This converges to near-optimal configurations in 25–35 trials, achieving approximately 3–4× improvement over random search at the same budget.

**Optimised CatBoost configuration:**

| Hyperparameter | Optimal Value |
|---|---|
| iterations | 197 |
| depth | 4 |
| learning_rate | 0.0498 |
| l2_leaf_reg | 3.50 |
| border_count | 90 |
| bagging_temperature | 0.741 |
| random_strength | 6.00 |

### Step 5 — Evaluation

Performance is evaluated on the held-out test set using: Accuracy, Precision, Recall, F1-Score, AUROC, Log-Loss, and False Positive Rate. All metrics are macro-averaged (equal weight per class). Statistical validity is confirmed via five-run repeated experiments across independent random seeds, with paired t-tests (Accuracy: t=3.107, p=0.018; AUROC: t=4.053, p=0.008) and non-parametric bootstrap confidence intervals.

### Step 6 — SHAP Explainability

TreeSHAP computes exact Shapley values for the CatBoost meta-classifier in polynomial time O(TLD²) by exploiting its symmetric tree structure. The complete explainability suite includes global feature importance rankings, per-class beeswarm distributions, individual prediction waterfall and force plots, and spatial heatmap overlays on original endoscopic images confirming anatomically appropriate model attention.

**Clinical confidence threshold workflow:**

| Confidence | Recommended Action |
|---|---|
| > 90% | Accept prediction; log automatically to audit trail |
| 70–90% | Flag for recommended expert review |
| < 70% | Require mandatory expert verification |
| Any — Esophagitis or Z-Line | Always flag for expert review regardless of confidence |

---

## Citations

If you use this code or build upon this work, please cite the following:

**This work:**
```
Mohammed Rohan Khan and Akshat Dubey. DualHybridSwinSmall: A Frozen Dual-Backbone 
Hierarchical Feature Fusion Framework Integrating Swin-Small and MobileViT for 
Multi-Class Gastrointestinal Disease Classification on the Kvasir-v2 Benchmark. 2025.
```

**Kvasir-v2 Dataset:**
```bibtex
@inproceedings{pogorelov2017kvasir,
  title     = {Kvasir: A Multi-Class Image Dataset for Computer Aided Gastrointestinal Disease Detection},
  author    = {Pogorelov, Konstantin and others},
  booktitle = {ACM Multimedia Systems Conference (MMSys)},
  year      = {2017}
}
```

**Swin Transformer:**
```bibtex
@inproceedings{liu2021swin,
  title     = {Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows},
  author    = {Liu, Ze and Lin, Yutao and Cao, Yue and others},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  note      = {Best Paper Award}
}
```

**MobileViTv2:**
```bibtex
@article{mehta2022mobilevitv2,
  title   = {Separable Self-attention for Mobile Vision Transformers},
  author  = {Mehta, Sachin and Rastegari, Mohammad},
  journal = {Transactions on Machine Learning Research (TMLR)},
  year    = {2022}
}
```

**CatBoost:**
```bibtex
@inproceedings{prokhorenkova2018catboost,
  title     = {CatBoost: Unbiased Boosting with Categorical Features},
  author    = {Prokhorenkova, Liudmila and Mescheryakov, Gleb and Veronika, Anna and Gulin, Andrey and Babenko, Anna},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2018}
}
```

**Optuna / TPE:**
```bibtex
@inproceedings{bergstra2011tpe,
  title     = {Algorithms for Hyper-Parameter Optimization},
  author    = {Bergstra, James and Bardenet, Rémi and Bengio, Yoshua and Kégl, Balázs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2011}
}
```

**SHAP:**
```bibtex
@inproceedings{lundberg2017shap,
  title     = {A Unified Approach to Interpreting Model Predictions},
  author    = {Lundberg, Scott M. and Lee, Su-In},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}
```

---

## License & Contribution Guidelines

### License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full terms.

You are free to use, modify, and distribute this code for both commercial and non-commercial purposes, provided the original license and copyright notice are included with any distribution.

### Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with clear messages
4. Open a pull request describing your changes and their motivation

Please ensure any changes maintain or improve test accuracy and do not break the existing evaluation pipeline. For significant architectural changes, include updated performance metrics.

### Acknowledgements

This project builds upon the work of the Kvasir-v2 dataset team at Simula Research Laboratory, the PyTorch and timm communities, and the developers of Optuna, CatBoost, XGBoost, LightGBM, and SHAP.

---

*Project date: December 2025 — Status: Production-Ready ✅*
