# MNIST and FashionMNIST Research Workflow

This project is a professional, notebook-first machine learning workflow for
MNIST and FashionMNIST. It is designed for college students learning how a
research-quality computer vision project is organized: data preparation, EDA,
preprocessing, baseline models, LeNet/AlexNet-style CNNs, training dynamics,
evaluation, error analysis, and interpretability.

## Project Layout

```text
data/
  raw/                 Original torchvision dataset files.
  processed/           Tensor and NumPy training formats.
  images/              PNG image exports grouped by split and label.
notebooks/
  00_problem_framing/
  01_EDA/
  02_preprocessing_augmentation/
  03_baselines/
  04_lenet_alexnet/
  05_training_pipeline/
  06_evaluation_error_analysis/
  07_interpretability_report/
src/
  mnist_fashion_research/
scripts/
  prepare_data.py
  generate_notebooks.py
  validate_project.py
outputs/
  figures/
  models/
  reports/
```

Each notebook step has separate `MNIST/` and `FashionMNIST/` folders.

## Quick Start

Prepare both datasets in raw, tensor, NumPy, and image formats:

```bash
python scripts/prepare_data.py
```

Regenerate all notebooks:

```bash
python scripts/generate_notebooks.py
```

Run a quick project validation:

```bash
python scripts/validate_project.py
```

## Research Workflow

The workflow is based on the supplied consultation links and standard computer
vision research practice:

1. Problem framing and reproducibility plan.
2. Dataset EDA with integrity checks, distributions, statistics, PCA structure,
   outliers, duplicates, and prototype similarity.
3. Preprocessing and augmentation decisions.
4. Baseline models to establish what simple methods can achieve.
5. LeNet and compact AlexNet-style CNN experiments.
6. Training pipeline with seeds, logging, checkpoints, and learning curves.
7. Evaluation with metrics, confusion matrices, error galleries, robustness
   checks, and calibration.
8. Interpretability and final research reporting.

The notebooks include Markdown explanations plus animations or visualizations
for the major concepts.

