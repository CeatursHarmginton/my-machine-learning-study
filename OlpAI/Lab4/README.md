# Advanced Linear Regression Activity Pack

This package contains a complete teaching activity set for advanced linear regression methods:

- Full Ordinary Least Squares (OLS)
- Backward selection using p-values
- Forward selection using p-values
- Ridge regression
- Lasso regression
- Statistical inference and post-selection inference caveats
- Bootstrap uncertainty for penalized models
- Visualization for model interpretability

## Main files

### Guides
- `docs/advanced_linear_regression_activity_guide_en.docx`: English instructor guide
- `docs/advanced_linear_regression_activity_guide_vi.docx`: Vietnamese instructor guide

### Notebooks
- `notebooks/advanced_linear_regression_activities_en.ipynb`: English notebook
- `notebooks/advanced_linear_regression_activities_vi.ipynb`: Vietnamese notebook

### Datasets
- `datasets/dataset_A_wide_housing.csv`: wide synthetic housing dataset
- `datasets/dataset_B_marketing_sales.csv`: medium-width synthetic marketing dataset
- `datasets/true_coefficients_dataset_A.csv`: known data-generating signals for Dataset A
- `datasets/true_coefficients_dataset_B.csv`: known data-generating signals for Dataset B
- `datasets/data_dictionary.csv`: column roles and descriptions

### Spreadsheet
- `model_comparison_template.xlsx`: Excel template for students to fill in method comparisons, feature-selection history, inference summaries, and visualization checklist

## Suggested workflow

1. Read the instructor guide.
2. Open the notebook and run Part A with Dataset A.
3. Have students fill out the Dataset A sheet in the Excel template.
4. Repeat the workflow with Dataset B.
5. Students submit a short report comparing model choice for prediction, interpretation, and statistical inference.

## Python packages used in the notebook

- numpy
- pandas
- matplotlib
- scipy
- statsmodels
- scikit-learn

The notebooks were syntax-checked and test-executed in the creation environment.
