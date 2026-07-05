# From MLP to Deep CNNs

This package contains the write-up and two notebooks for a 3-part practical assignment in Computer Vision with Deep Learning.

## Files

- `thuc_hanh_6.docx`: Assignment write-up in Vietnamese.
- `notebooks/cv_weeks1_3_starter.ipynb`: Starter notebook with TODO sections.
- `notebooks/cv_weeks1_3_solution.ipynb`: Reference solution notebook with PyTorch pipeline implementation.
- `requirements.txt`: Suggested Python packages.

## Recommended workflow

1. Open the Word document and read the assignment requirements.
2. Open the starter notebook.
3. Run with `FAST_DEV_RUN=True` to test code quickly.
4. Complete the TODO sections.
5. Run each experiment configuration and export results to CSV.
6. Prepare the final report with tables, plots, confusion matrices, and analysis.

## Dataset notes

- Phase 1 uses MNIST or Fashion-MNIST.
- Phase 2 uses CIFAR-10.
- Phase 3 expects Tiny ImageNet or ImageNet-10 in ImageFolder format:

```text
DATASET_ROOT/
├── train/
│   ├── class_001/
│   └── ...
├── val/
│   ├── class_001/
│   └── ...
└── test/
    ├── class_001/
    └── ...
```

The solution notebook includes an `IMAGEFOLDER_ROOT` variable that should be changed before running Phase 3.
