# Consultation Synthesis

Some supplied consultation links were readable in the browser session, while
the Gemini links reported that the shared links did not exist and one Grok EDA
page did not expose readable text. The readable consultations converged on the
same professional process.

## EDA Decisions

Professional EDA should include:

- Dataset overview: shape, split sizes, class names, image size, dtype, pixel
  range, and label integrity.
- Class balance: counts, percentages, imbalance ratio, and whether class
  weighting is needed.
- Pixel statistics: global histogram, per-class mean intensity, standard
  deviation, sparsity, and active-pixel ratio.
- Visual inspection: sample grids, class prototypes, per-class mean images, and
  per-class standard deviation maps.
- Data quality: corrupted/all-zero images, duplicates, ambiguous examples, and
  possible train/test leakage checks.
- Geometry: PCA explained variance, 2D PCA scatter, optional t-SNE/UMAP, and
  class prototype distance/similarity matrices.
- Outliers: unusual examples using PCA features and anomaly detection.
- Research implications: what the EDA suggests about normalization,
  augmentation, model capacity, and expected failure modes.

## Modeling Workflow Decisions

The later workflow should include:

- Reproducibility setup with fixed seeds, explicit folders, and saved configs.
- Separate preprocessing and augmentation, because normalization makes data
  usable while augmentation tests robustness.
- Baselines before CNNs: majority class, nearest centroid, logistic regression,
  MLP, and a simple CNN where useful.
- Main CNNs: LeNet as the historically appropriate small CNN; a compact
  AlexNet-style model adapted to 28x28 grayscale images.
- Training dynamics: train/validation curves, learning-rate choices,
  overfitting/underfitting checks, checkpoints, and experiment logs.
- Evaluation: accuracy, macro/micro F1, confusion matrix, per-class accuracy,
  error galleries, confidence analysis, calibration, and robustness to
  rotations/noise/shifts.
- Interpretability: saliency, occlusion sensitivity, feature-map inspection,
  and concise research reporting.

## Animation Choices

Animations are included where they teach an evolving process:

- Sample carousels per class.
- Cumulative class mean formation.
- Threshold sweeps that reveal foreground sparsity.
- Augmentation sequences.
- Learning curves built epoch by epoch.
- Convolution sliding-window demonstrations.
- Robustness/error review sequences.
- Occlusion sensitivity sweeps.

When animation is less informative than a static display, notebooks use plots,
heatmaps, tables, or image grids instead.

