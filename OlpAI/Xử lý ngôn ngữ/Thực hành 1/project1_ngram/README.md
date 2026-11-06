# NLP Project 1 — N-gram Language Model

Build a classical statistical N-gram language model from scratch on Jane Austen's *Emma* (NLTK Gutenberg), then evaluate with held-out perplexity and generate text by sampling.

## Research question

Does using a longer context (a larger $N$) always create a better language model?

## Project layout

```
project1_ngram/
├── src/ngram_model.py          # Reusable model functions
├── notebooks/experiments.ipynb # Full walkthrough + experiments
├── outputs/
│   ├── results.csv             # Required summary table
│   ├── perplexity_grid.csv     # Full N × k grid
│   ├── generated_samples.txt   # Sampled text for N=1..4
│   └── perplexity_sparsity.png # Plots
└── README.md
```

## Setup

```bash
pip install nltk numpy pandas matplotlib jupyter
```

Open and run `notebooks/experiments.ipynb` top to bottom. The first code cell downloads NLTK resources (`gutenberg`, `punkt`, `punkt_tab`).

## Method (brief)

1. **Split** sentences 80% / 10% / 10% in corpus order.
2. **Vocabulary** from training only; words with count ≤ 1 → `<UNK>`.
3. **Count** unigram…4-gram with `<s>` / `</s>` padding.
4. **MLE** and **add-$k$** smoothing ($k \in \{0.01, 0.1, 1.0\}$).
5. **Tune $k$** on validation; report test perplexity once.
6. **Predict** top-5 next words; **generate** by sampling.

Optional: linear interpolation of unigram/bigram/trigram, with $\lambda$ tuned on validation.

## Reproducibility

- Deterministic train/valid/test split (no shuffle).
- Generation uses fixed random seeds (default `seed=7`).
- All counts and vocabulary come from the training split only.

## Interactive web demo

Open `web/index.html` in a browser (double-click is fine; data is embedded in `corpus-data.js`).

The N-gram model is trained on the **entire** *Emma* novel (~192k tokens). Three paragraphs are shown only as an inspection sample.

Features:

- Buttons to choose \(N \in \{1,2,3,4\}\)
- Click a word to see full-corpus MLE probability, history highlighting, and the count equation
- Next-word playground with top-5 suggestions from the full-corpus model

Rebuild counts after changing the export script:

```bash
python web/build_corpus_data.py
```
