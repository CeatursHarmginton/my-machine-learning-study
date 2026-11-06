# NLP Project 4 — Movie Review Sentiment Analysis with N-grams

Binary sentiment classification on the Stanford Large Movie Review Dataset (IMDb) using Bag-of-Words, TF-IDF, word N-grams, Logistic Regression, and Multinomial Naive Bayes.

## Research question

Do bigram/trigram features improve sentiment classification beyond unigram Bag-of-Words, and does TF-IDF change the answer?

## Project layout

```text
project4_sentiment/
├── data/aclImdb/              # Stanford IMDb dataset (download once)
├── src/
│   ├── load_data.py           # Load reviews, clean text, create validation split
│   └── train.py               # Train configs A–F and evaluate final test model
├── notebooks/
│   └── experiments.ipynb      # Guided experiments with Markdown explanations
├── outputs/
│   ├── model_results.csv
│   ├── confusion_matrix.png
│   └── ...
├── report.pdf
├── requirements.txt
└── README.md
```

## Environment

```bash
python -m pip install -r requirements.txt
```

## Dataset

Download and extract the official archive into `data/`:

```bash
# from project4_sentiment/
curl -L -o data/aclImdb_v1.tar.gz https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf data/aclImdb_v1.tar.gz -C data
```

Expected folders: `data/aclImdb/train/{pos,neg}` and `data/aclImdb/test/{pos,neg}`.

## Reproduce experiments

### Option A — Jupyter notebook (recommended for learning)

```bash
cd notebooks
jupyter notebook experiments.ipynb
```

Run all cells top to bottom. The notebook explains each step in Markdown, keeps comments in code, compares configurations A–F on validation data, then evaluates the chosen model once on the official test set.

### Option B — CLI training script

```bash
cd src
python train.py
```

This writes `outputs/model_results.csv`, `outputs/confusion_matrix.png`, and related artifacts.

## Experimental protocol

1. Load the official 25k/25k train/test split.
2. Create a stratified 80/20 train/validation split from the official training set only.
3. Apply light cleaning (HTML `<br>` removal, whitespace normalization). Do **not** remove stopwords by default.
4. Compare fixed configurations A–F with shared `min_df`, `max_features`, and random seed.
5. Select the best model on **validation** metrics.
6. Retrain on all official training reviews and evaluate **once** on the official test set.

## Required configurations

| ID | Representation | N-gram range | Classifier |
|----|----------------|--------------|------------|
| A | BoW counts | (1,1) | Logistic Regression |
| B | BoW counts | (1,2) | Logistic Regression |
| C | TF-IDF | (1,1) | Logistic Regression |
| D | TF-IDF | (1,2) | Logistic Regression |
| E | TF-IDF | (1,3) | Logistic Regression |
| F | BoW counts | (1,2) | Multinomial NB |

## Notes on leakage

- Fit vectorizers only on training text (`fit_transform`), then `transform` validation/test.
- Do not choose hyperparameters using the official test set.
- Prefer `sklearn.pipeline.Pipeline` so the vectorizer and classifier stay paired.

## References

- Maas, A. et al. (2011). [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
