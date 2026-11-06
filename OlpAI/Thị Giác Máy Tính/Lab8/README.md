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
  MNIST/
    combined_pipeline_MNIST.ipynb
  FashionMNIST/
    combined_pipeline_FashionMNIST.ipynb
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

Each dataset has one combined pipeline notebook. The combined notebooks omit
the original problem framing and EDA steps, then continue with preprocessing,
augmentation, model training, checkpointing, and evaluation.

## Quick Start

Prepare both datasets in raw, tensor, NumPy, and image formats:

```bash
python scripts/prepare_data.py
```

Regenerate the combined notebooks:

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

1. Preprocessing and augmentation decisions.
2. MLP baseline experiments with hidden layer sizes of 64, 128, and 256.
3. LeNet and compact AlexNet-style CNN experiments with configurable
   convolution channel lists.
4. Training with rotation, affine, stratified split, and hyperparameter sweeps.
5. Checkpoint saving for trained models.
6. Evaluation with metrics, confusion matrices, parameter comparisons, error
   galleries, training curves, and plots.

The notebooks include Markdown explanations plus visualizations for the major
pipeline comparisons.
