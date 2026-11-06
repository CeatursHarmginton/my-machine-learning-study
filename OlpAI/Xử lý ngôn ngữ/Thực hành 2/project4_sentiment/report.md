# NLP Project 4 Report  
Movie Review Sentiment Analysis with N-grams (BoW / TF-IDF)

## 1. Overview

This project builds a binary sentiment classifier for IMDb movie reviews (positive vs negative). Unigrams, bigrams, and trigrams are used as **classification features** (not as a next-word language model). We compare raw Bag-of-Words counts with TF-IDF weighting under a fixed preprocessing and train/validation protocol, then evaluate the selected model once on the official test set.

**Research question.** Do bigram/trigram features improve sentiment classification beyond unigram Bag-of-Words, and does TF-IDF change the answer?

## 2. Data and protocol

- **Dataset:** Stanford Large Movie Review Dataset (`aclImdb`), 25,000 train + 25,000 test reviews, balanced labels.
- **Validation:** stratified 80/20 split of the official training set (`random_state=42`).
- **Cleaning:** HTML entity unescape, `<br>` removal, whitespace normalization. Stopwords were **not** removed, to preserve negation cues such as *not* / *never*.
- **Leakage control:** vectorizers are fit on training text only; model selection uses validation metrics only; the official test set is used once for the final model.

## 3. Configurations

| ID | Representation | N-gram range | Classifier |
|----|----------------|--------------|------------|
| A | BoW counts | (1,1) | Logistic Regression |
| B | BoW counts | (1,2) | Logistic Regression |
| C | TF-IDF | (1,1) | Logistic Regression |
| D | TF-IDF | (1,2) | Logistic Regression |
| E | TF-IDF | (1,3) | Logistic Regression |
| F | BoW counts | (1,2) | Multinomial Naive Bayes |

Shared settings: `min_df=2`, `max_features=150000`, `sublinear_tf=True` for TF-IDF, Logistic Regression `max_iter=1000`.

## 4. Validation results

| ID | # features | Val accuracy | Val precision | Val recall | Val F1 |
|----|------------|--------------|---------------|------------|--------|
| A | 40,909 | 0.8742 | 0.8690 | 0.8812 | 0.8751 |
| B | 150,000 | 0.8864 | 0.8824 | 0.8916 | 0.8870 |
| C | 40,909 | 0.8798 | 0.8713 | 0.8912 | 0.8812 |
| D | 150,000 | 0.8864 | 0.8773 | 0.8984 | 0.8877 |
| E | 150,000 | **0.8878** | 0.8789 | 0.8996 | **0.8891** |
| F | 150,000 | 0.8654 | 0.8748 | 0.8528 | 0.8637 |

**Selected model:** configuration **E** (TF-IDF, n-gram range 1–3, Logistic Regression), best validation F1.

## 5. Official test evaluation

After retraining E on all 25,000 official training reviews:

| Metric | Test score |
|--------|------------|
| Accuracy | **0.8989** |
| Precision | 0.8948 |
| Recall | 0.9042 |
| F1 | 0.8994 |

The confusion matrix is saved in `outputs/confusion_matrix.png`.

## 6. Feature inspection (Logistic Regression coefficients)

Strong positive unigrams include *great*, *excellent*, *wonderful*, *perfect*, *amazing*.  
Strong negative unigrams include *bad*, *worst*, *awful*, *boring*, *waste*.

Useful multi-word features among the strongest weights include:

- **the best** / **of the best** (positive) — conventional endorsement phrases.
- **must see**, **very good**, **well worth**, **my favorite** (positive) — collocations and intensifiers.
- **the worst** / **of the worst** (negative) — conventional pejorative phrases.
- **waste of**, **not even**, **not worth**, **at best** (negative) — negation and idiomatic negative evaluation.

These phrases illustrate why N-grams matter: polarity often comes from the combination, not from either word alone.

## 7. Error analysis (qualitative)

Typical remaining errors for the classical model include:

1. **Sarcasm / ironic praise** that uses positive vocabulary in a negative review.
2. **Mixed sentiment** (e.g., praise for acting + criticism of plot).
3. **Plot-heavy reviews** whose dark vocabulary looks negative even when the rating is positive.
4. **Long-range negation** beyond local bigram/trigram windows.
5. **Proper names / franchise terms** that correlate with labels in training data but are not true sentiment cues.

Detailed excerpts are examined in `notebooks/experiments.ipynb` (Section 13).

## 8. Discussion questions

### 8.1 Did adding bigrams improve validation performance?

**Yes.**  

- BoW: A → B accuracy rose from **0.8742 → 0.8864** (+1.22 points); F1 from 0.8751 → 0.8870.  
- TF-IDF: C → D accuracy rose from **0.8798 → 0.8864** (+0.66 points); F1 from 0.8812 → 0.8877.

Phrase features help encode local order such as negation and collocations that unigrams miss.

### 8.2 Two bigrams whose meaning is not obvious from either unigram alone

1. **waste of** — *waste* is negative and *of* is neutral; together they form an idiomatic complaint (*waste of time/money*) stronger than either token alone.  
2. **not worth** — *not* flips polarity and *worth* is often positive; the bigram encodes a negative recommendation that unigram bag-of-words can scramble.

Related learned phrases: **the worst**, **must see**, **at best**, **not even**.

### 8.3 How did TF-IDF change performance relative to raw counts?

TF-IDF helped modestly on unigrams (**A 0.8742 → C 0.8798** accuracy) and was essentially tied with BoW on unigram+bigram (**B and D both 0.8864** accuracy), with a small F1 edge for TF-IDF D (0.8877 vs 0.8870).  

Interpretation: weighting reduces the dominance of very common terms, but once informative bigrams are included, Logistic Regression already uses count features effectively. TF-IDF is helpful, not transformative.

### 8.4 Did trigrams help?

Slightly. **D → E**: accuracy **0.8864 → 0.8878**, F1 **0.8877 → 0.8891**.  

**Trade-off:** trigrams add richer local context, but also increase sparsity and redundancy. With `max_features=150000`, many rare trigrams compete for slots with useful unigrams/bigrams. Gains exist but are small; more features is not automatically better.

### 8.5 Which reviews are still difficult?

Reviews with sarcasm, conflicting opinions inside one text, heavy plot description, subtle or long-distance negation, and spurious name/genre correlations remain difficult for bag-of-N-grams linear models. These cases often need discourse-level or contextual representations beyond sparse N-gram features.

## 9. Conclusion

Bigram features clearly improve over unigram BoW. TF-IDF gives a smaller additional boost, mainly for unigrams. Trigrams add a marginal gain under a fixed feature budget. The selected TF-IDF (1,3) + Logistic Regression model reaches about **89.9% accuracy** on the official IMDb test set. Remaining errors highlight the limits of local N-gram features for nuanced sentiment.

## References

- Maas, A. et al. (2011). Large Movie Review Dataset. https://ai.stanford.edu/~amaas/data/sentiment/
- scikit-learn documentation for `CountVectorizer` and `TfidfVectorizer`.
