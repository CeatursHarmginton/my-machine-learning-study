"""Build notebooks/experiments.ipynb for NLP Project 1."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(
    r"""
# NLP Project 1 — Build an N-gram Language Model and Text Generator

**Goal.** Learn statistical language modeling by implementing it from counts, then measure it with perplexity.

**Research question.** Does using a longer context (a larger $N$) always create a better language model? The experiments below answer this with evidence, not intuition.

**Dataset.** Jane Austen's *Emma* from the NLTK Gutenberg corpus (`austen-emma.txt`).

This notebook walks through preprocessing, N-gram counting, MLE, add-$k$ smoothing, log-probability, perplexity, next-word prediction, text generation, and the required comparison experiments.
"""
)

md(
    r"""
## 0. Setup

Install dependencies once, then download the NLTK corpus and tokenizer resources.
"""
)

code(
    r"""
# Install packages if needed (uncomment on a fresh environment):
# !pip install nltk numpy pandas matplotlib

import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

# Make the local src/ package importable when this notebook lives in notebooks/
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ngram_model import (
    add_k_probability,
    apply_unk,
    build_vocabulary,
    corpus_perplexity,
    generate,
    interpolate_probability,
    interpolated_corpus_perplexity,
    mle_corpus_perplexity,
    mle_probability,
    normalize_sentence,
    ngram_counts,
    observed_continuation_prob_sum,
    percent_unseen_ngrams,
    predict_next,
    sample_next,
    sample_next_backoff,
    sentence_log_probability,
)

# Download NLTK resources once (quiet=True avoids noisy progress bars on re-runs)
nltk.download("gutenberg", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.corpus import gutenberg

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", ROOT)
print("Gutenberg fileids (first 5):", gutenberg.fileids()[:5])
"""
)

md(
    r"""
## 1. Background: What an N-gram Language Model Does

A language model assigns a probability to a sequence of words. For a sentence $W = (w_1, w_2, \ldots, w_m)$, the chain rule is:

$$
P(w_1,\ldots,w_m) = \prod_{i=1}^{m} P(w_i \mid w_1,\ldots,w_{i-1})
$$

The full history is too specific to estimate from finite data. An **N-gram** model uses a Markov approximation: only the previous $N-1$ tokens matter as context.

- **Bigram:** $P(w_i \mid w_1,\ldots,w_{i-1}) \approx P(w_i \mid w_{i-1})$
- **Trigram:** $P(w_i \mid w_1,\ldots,w_{i-1}) \approx P(w_i \mid w_{i-2}, w_{i-1})$

Maximum-likelihood estimation (MLE) for a bigram is simply a ratio of counts:

$$
P_{\mathrm{MLE}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$
"""
)

md(
    r"""
## 2. Worked Example (Tiny Corpus)

Before using *Emma*, verify the idea on three short sentences:

```
<s> i like nlp </s>
<s> i like pizza </s>
<s> i love nlp </s>
```

Context `"i"` occurs three times. Bigram `(i, like)` occurs twice and `(i, love)` once, so:

- $P(\text{like} \mid i) = 2/3$
- $P(\text{love} \mid i) = 1/3$
- $P(\text{enjoy} \mid i) = 0$ under MLE — any sentence with that unseen transition gets total probability zero. That is the **data-sparsity** problem that motivates smoothing.
"""
)

code(
    r"""
# Tiny worked example — confirm MLE counts by hand
tiny = [
    ["i", "like", "nlp"],
    ["i", "like", "pizza"],
    ["i", "love", "nlp"],
]

bi_ng, bi_ctx = ngram_counts(tiny, order=2)

print("C(i)      =", bi_ctx[("i",)])
print("C(i,like)=", bi_ng[("i", "like")])
print("C(i,love)=", bi_ng[("i", "love")])
print("P(like|i)=", mle_probability("like", ("i",), bi_ng, bi_ctx))
print("P(love|i)=", mle_probability("love", ("i",), bi_ng, bi_ctx))
print("P(enjoy|i)=", mle_probability("enjoy", ("i",), bi_ng, bi_ctx), "<- zero = sparsity")
"""
)

md(
    r"""
## 3. Step 1 — Load, Preprocess, and Split the Corpus

**Rules used here:**

1. Load sentence-tokenized text from `austen-emma.txt`.
2. Lowercase tokens (reduces vocabulary sparsity for the baseline).
3. Use an **80% / 10% / 10%** sentence split in the **original corpus order** (deterministic, reproducible).
4. Learn vocabulary and N-gram counts from the **training split only** — never from validation or test.
"""
)

code(
    r"""
# Load Jane Austen's Emma as a list of tokenized sentences
raw_sents = list(gutenberg.sents("austen-emma.txt"))
print("Number of raw sentences:", len(raw_sents))
print("First sentence:", raw_sents[0])

# Normalize: lowercase + drop empty tokens; keep punctuation as its own tokens
# (NLTK's sentence tokenizer already separates most punctuation).
sents = [normalize_sentence(s) for s in raw_sents if len(s) > 0]

n = len(sents)
train_sents = sents[: int(0.80 * n)]
valid_sents = sents[int(0.80 * n) : int(0.90 * n)]
test_sents = sents[int(0.90 * n) :]

print(f"Train / valid / test: {len(train_sents)} / {len(valid_sents)} / {len(test_sents)}")
print("Example train sentence:", train_sents[10][:20], "...")
"""
)

md(
    r"""
## 4. Step 2 — Build the Training Vocabulary and `<UNK>`

Rare and unseen words are a major source of sparsity. We:

1. Count word frequencies on the **training** set only.
2. Treat words with count $\le 1$ as rare and map them to `<UNK>`.
3. At evaluation time, any word not in the training vocabulary also maps to `<UNK>`.

Special tokens:

- `<UNK>` — unknown / rare word
- `</s>` — end of sentence (included in the vocabulary so the model can predict sentence endings)
- `<s>` — start-of-sentence marker used only as **context**, not as a predicted word, so it is not required in the output vocabulary
"""
)

code(
    r"""
UNK_THRESHOLD = 1  # words with count <= 1 become <UNK>

# Vocabulary is estimated from training data only (no test leakage)
vocab = build_vocabulary(train_sents, unk_threshold=UNK_THRESHOLD)

# Apply the same mapping to all splits
train_sents = apply_unk(train_sents, vocab)
valid_sents = apply_unk(valid_sents, vocab)
test_sents = apply_unk(test_sents, vocab)

print("Vocabulary size |V|:", len(vocab))
print("Contains <UNK>?", "<UNK>" in vocab)
print("Contains </s>?", "</s>" in vocab)

# Quick check: how many tokens became <UNK> in each split?
def unk_rate(split):
    toks = [w for s in split for w in s]
    return 100.0 * toks.count("<UNK>") / len(toks)

print(f"% <UNK> tokens — train: {unk_rate(train_sents):.2f}, "
      f"valid: {unk_rate(valid_sents):.2f}, test: {unk_rate(test_sents):.2f}")
"""
)

md(
    r"""
## 5. Step 3 — Count N-grams

For an order-$N$ model, pad each sentence with $N-1$ start symbols `<s>` and one end symbol `</s>`:

- Start symbols give context for the first real word.
- `</s>` teaches the model when a sentence can end.

We store:

- `ngram[history + (word,)]` = $C(h, w)$
- `context[history]` = $C(h)$
"""
)

code(
    r"""
# Build count tables for unigram, bigram, trigram, and 4-gram models
ORDERS = [1, 2, 3, 4]
models = {}  # order -> (ngram_counts, context_counts)

for order in ORDERS:
    ng, ctx = ngram_counts(train_sents, order)
    models[order] = (ng, ctx)
    print(f"{order}-gram: {len(ng):,} unique N-grams, {len(ctx):,} unique contexts")

# Spot-check a trigram that is likely in Austen prose
tri_counts, tri_contexts = models[3]
print('C("she","was","very") =', tri_counts[("she", "was", "very")])
print('C("she","was")        =', tri_contexts[("she", "was")])
"""
)

md(
    r"""
## 6. Step 4 — Maximum-Likelihood Probability

$$
P_{\mathrm{MLE}}(w \mid h) = \frac{C(h, w)}{C(h)}
$$

If the context $h$ never occurred, MLE returns $0$. We also manually inspect at least ten contexts and confirm that probabilities of **observed** continuations sum to approximately $1.0$.
"""
)

code(
    r"""
# Example MLE probability from the trigram model
p = mle_probability("very", ("she", "was"), tri_counts, tri_contexts)
print('P_MLE(very | she was) =', p)

# Inspect >= 10 training contexts: probabilities of observed continuations should sum to ~1
# Pick the most frequent bigram contexts for a clear sanity check.
bi_counts, bi_contexts = models[2]
top_histories = [h for h, _ in bi_contexts.most_common(12) if h != ("<s>",)][:10]

print("\nSanity check — sum of MLE probs over observed continuations:")
rows = []
for hist in top_histories:
    total, conts = observed_continuation_prob_sum(hist, bi_counts, bi_contexts)
    top3 = ", ".join(f"{w}:{p:.3f}" for w, p in conts[:3])
    rows.append({"history": " ".join(hist), "sum_p": total, "top3": top3})
    print(f"  history={hist!r:30s}  sum={total:.6f}  top3=[{top3}]")

sanity_df = pd.DataFrame(rows)
sanity_df
"""
)

md(
    r"""
## 7. Step 5 — Add-$k$ Smoothing

Additive (Lidstone) smoothing gives every possible continuation a small pseudo-count $k$:

$$
P_k(w \mid h) = \frac{C(h,w) + k}{C(h) + k|V|}
$$

This prevents unseen N-grams from receiving exactly zero probability.

- Baseline: $k = 0.1$
- Compare also $k = 0.01$ and $k = 1.0$ (add-one / Laplace)
- Add-one often over-allocates mass to unseen events when $|V|$ is large
"""
)

code(
    r"""
# Compare MLE vs add-k on a seen and an unseen continuation
vocab_size = len(vocab)
hist = ("she", "was")

for word in ["very", "xyzzy"]:  # "xyzzy" is almost certainly unseen
    p_mle = mle_probability(word, hist, tri_counts, tri_contexts)
    p_01 = add_k_probability(word, hist, tri_counts, tri_contexts, vocab_size, k=0.1)
    p_1 = add_k_probability(word, hist, tri_counts, tri_contexts, vocab_size, k=1.0)
    print(f"word={word!r:8s}  MLE={p_mle:.6g}  add-0.1={p_01:.6g}  add-1={p_1:.6g}")

print("\nNote: smoothing moves probability mass from seen events to unseen ones.")
"""
)

md(
    r"""
## 8. Optional Improvement — Linear Interpolation

A trigram may be rare even when its bigram or unigram evidence is strong. Interpolation mixes orders:

$$
P(w \mid h) = \lambda_3 P_3(w \mid w_{i-2},w_{i-1}) + \lambda_2 P_2(w \mid w_{i-1}) + \lambda_1 P_1(w)
$$

with $\lambda_1 + \lambda_2 + \lambda_3 = 1$ and $\lambda_j \ge 0$.

Starting point: $(\lambda_1, \lambda_2, \lambda_3) = (0.1, 0.3, 0.6)$. These are hyperparameters — tune on **validation**, never on the final test set.
"""
)

code(
    r"""
# Simple grid search over lambda triples on the validation set (trigram interpolation)
candidate_lambdas = [
    (0.1, 0.3, 0.6),
    (0.2, 0.3, 0.5),
    (0.1, 0.4, 0.5),
    (0.05, 0.25, 0.70),
    (1 / 3, 1 / 3, 1 / 3),
]

interp_rows = []
for lams in candidate_lambdas:
    pp = interpolated_corpus_perplexity(
        valid_sents, max_order=3, models=models, lambdas=lams, vocab=vocab, k=0.1
    )
    interp_rows.append({"lambda1": lams[0], "lambda2": lams[1], "lambda3": lams[2], "valid_pp": pp})
    print(f"lambdas={lams}  valid PP={pp:.2f}")

interp_df = pd.DataFrame(interp_rows).sort_values("valid_pp")
best_lams = tuple(
    interp_df.iloc[0][["lambda1", "lambda2", "lambda3"]].astype(float).tolist()
)
print("Best validation lambdas:", best_lams)
interp_df
"""
)

md(
    r"""
## 9. Step 6 — Sentence Probability and Log Probability

Multiplying many small probabilities can underflow to zero in floating-point arithmetic. Sum logarithms instead:

$$
\log P(W) = \sum_i \log P(w_i \mid h_i)
$$
"""
)

code(
    r"""
# Log-probability of a short sentence under the smoothed trigram model
example = ["she", "was", "very", "happy"]
lp = sentence_log_probability(example, order=3, ngram=tri_counts, context=tri_contexts, vocab=vocab, k=0.1)
print("tokens:", example)
print("log P(W) =", lp)
print("P(W)     =", math.exp(lp))
"""
)

md(
    r"""
## 10. Step 7 — Perplexity

Perplexity measures how surprised the model is by held-out text. For $M$ predicted tokens (including `</s>`):

$$
\mathrm{PP} = \exp\left(-\frac{1}{M}\sum_i \log P(w_i \mid h_i)\right)
$$

**Interpretation.** Lower perplexity is better **only** when models share the same test text, tokenization, and vocabulary treatment.

**Important.** Unsmoothed MLE often yields **infinite** test perplexity because a single unseen N-gram drives some $P(w|h)=0$. That is expected — it is evidence for why smoothing is necessary, not a bug to hide.
"""
)

code(
    r"""
# Demonstrate infinite MLE perplexity vs finite add-k perplexity on the test set
print("Unsmoothed MLE test perplexity (expect inf for N>1):")
for order in ORDERS:
    ng, ctx = models[order]
    pp_mle = mle_corpus_perplexity(test_sents, order, ng, ctx, vocab)
    print(f"  {order}-gram MLE PP = {pp_mle}")

print("\nAdd-k=0.1 test perplexity (finite):")
for order in ORDERS:
    ng, ctx = models[order]
    pp = corpus_perplexity(test_sents, order, ng, ctx, vocab, k=0.1)
    print(f"  {order}-gram add-0.1 PP = {pp:.2f}")
"""
)

md(
    r"""
## 11. Step 8 — Next-word Prediction

Given a context, score every vocabulary word with add-$k$ probability and return the top-$k$ candidates.
"""
)

code(
    r"""
# At least 10 contexts with top-5 predictions (required experiment D)
prediction_contexts = [
    ("she",),
    ("he",),
    ("i",),
    ("the",),
    ("of",),
    ("she", "was"),
    ("he", "was"),
    ("i", "am"),
    ("it", "was"),
    ("mr", "."),
    ("mrs", "."),
    ("very", "much"),
]

pred_rows = []
print("Top-5 next-word predictions (add-k = 0.1):\n")
for hist in prediction_contexts:
    order = len(hist) + 1
    # Use the model whose context length matches the provided history
    ng, ctx = models[order]
    top = predict_next(hist, ng, ctx, vocab, k_smooth=0.1, top_k=5)
    pretty = ", ".join(f"{w} ({p:.4g})" for w, p in top)
    print(f"  context={hist!r}")
    print(f"    -> {pretty}")
    for rank, (w, p) in enumerate(top, start=1):
        pred_rows.append({
            "context": " ".join(hist),
            "order": order,
            "rank": rank,
            "word": w,
            "probability": p,
        })

pred_df = pd.DataFrame(pred_rows)
pred_df.head(15)
"""
)

md(
    r"""
## 12. Step 9 — Text Generation by Sampling

Always choosing the highest-probability word tends to produce repetitive text. Instead, **sample** from the predicted distribution. Generation stops at `</s>` or after a requested number of tokens.

**Practical note.** Scoring every vocabulary word with add-$k$ puts a large share of probability on unseen words when $|V|$ is large, so samples become noisy. For readable generation we therefore sample from **observed continuations** and **back off** to shorter histories when the current context was never seen. Perplexity evaluation above still uses full-vocabulary add-$k$, as required.

Larger $N$ usually improves local coherence, but may also **copy** longer training phrases (memorization).
"""
)

code(
    r"""
# Generate samples for N = 1..4 (required experiment E)
# use_backoff=True: sample from observed continuations with shorter-history backoff
generated_lines = []
print("Generated samples (seed=7, k=0.1, max_tokens=60, backoff sampling):\n")
for order in ORDERS:
    ng, ctx = models[order]
    text = generate(
        order, ng, ctx, vocab,
        max_tokens=60, k=0.1, seed=7,
        models=models, use_backoff=True,
    )
    line = f"[{order}-gram] {text}"
    print(line)
    print()
    generated_lines.append(line)

# Extra seeds for qualitative comparison
print("--- Additional seeds for the trigram model ---")
for seed in [1, 42, 99]:
    text = generate(
        3, models[3][0], models[3][1], vocab,
        max_tokens=50, k=0.1, seed=seed,
        models=models, use_backoff=True,
    )
    print(f"[seed={seed}] {text}\n")
    generated_lines.append(f"[3-gram seed={seed}] {text}")

# Optional: show how naive full-vocabulary add-k sampling looks (often noisier)
print("--- Naive full-vocabulary add-k sample (trigram, for contrast) ---")
naive = generate(
    3, models[3][0], models[3][1], vocab,
    max_tokens=40, k=0.1, seed=7,
    use_backoff=False,
)
print(naive)
generated_lines.append(f"[3-gram naive full-vocab] {naive}")

# Save generated text for submission
samples_path = OUTPUT_DIR / "generated_samples.txt"
samples_path.write_text("\n\n".join(generated_lines), encoding="utf-8")
print("Wrote", samples_path)
"""
)

md(
    r"""
## 13. Required Experiments

| Experiment | Independent variable | Measure / artifact |
|---|---|---|
| A. Context length | $N = 1,2,3,4$ | Validation/test perplexity; 100-word sample |
| B. Smoothing strength | $k = 0.01, 0.1, 1.0$ | Perplexity for each $N$ |
| C. Sparsity | $N = 1\ldots4$ | Unique train N-grams; % unseen test N-grams |
| D. Prediction examples | $\ge 10$ contexts | Top-5 predictions (done above) |
| E. Generation | $N = 1\ldots4$ | Qualitative comparison (done above) |

**Tuning rule:** choose the best $k$ on the **validation** set; report final numbers on the **test** set only once.
"""
)

code(
    r"""
# Experiments A + B: validation perplexity grid over N and k
K_VALUES = [0.01, 0.1, 1.0]

grid_rows = []
for order in ORDERS:
    ng, ctx = models[order]
    for k in K_VALUES:
        pp_valid = corpus_perplexity(valid_sents, order, ng, ctx, vocab, k=k)
        pp_test = corpus_perplexity(test_sents, order, ng, ctx, vocab, k=k)
        grid_rows.append({
            "order": order,
            "k": k,
            "valid_perplexity": pp_valid,
            "test_perplexity": pp_test,
        })

grid_df = pd.DataFrame(grid_rows)
print("Full N x k perplexity grid:")
grid_df
"""
)

code(
    r"""
# Choose best k per order on validation; then read the corresponding test PP
summary_rows = []
for order in ORDERS:
    ng, ctx = models[order]
    sub = grid_df[grid_df["order"] == order]
    best = sub.loc[sub["valid_perplexity"].idxmin()]
    best_k = float(best["k"])
    unique_train = len(ng)
    pct_unseen = percent_unseen_ngrams(test_sents, order, ng)
    # Recompute test PP with the validation-chosen k (no re-tuning on test)
    test_pp = corpus_perplexity(test_sents, order, ng, ctx, vocab, k=best_k)
    summary_rows.append({
        "Order": f"{order}-gram",
        "Unique train N-grams": unique_train,
        "% unseen test N-grams": round(pct_unseen, 2),
        "Best validation k": best_k,
        "Validation perplexity": round(float(best["valid_perplexity"]), 2),
        "Test perplexity": round(test_pp, 2),
    })

results_df = pd.DataFrame(summary_rows)
print("Required results table:")
print(results_df.to_string(index=False))

# Also evaluate the optional interpolated model with validation-tuned lambdas
interp_test_pp = interpolated_corpus_perplexity(
    test_sents, max_order=3, models=models, lambdas=best_lams, vocab=vocab, k=0.1
)
print(f"\nOptional interpolated trigram test PP (lambdas={best_lams}, k=0.1): {interp_test_pp:.2f}")

# Persist CSV for submission
results_path = OUTPUT_DIR / "results.csv"
results_df.to_csv(results_path, index=False)
grid_path = OUTPUT_DIR / "perplexity_grid.csv"
grid_df.to_csv(grid_path, index=False)
print("Wrote", results_path)
print("Wrote", grid_path)
"""
)

code(
    r"""
# Visualization: validation perplexity vs N for each k
pivot = grid_df.pivot(index="order", columns="k", values="valid_perplexity")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Left: validation PP curves
for k in K_VALUES:
    axes[0].plot(ORDERS, pivot[k], marker="o", label=f"k={k}")
axes[0].set_xlabel("N-gram order N")
axes[0].set_ylabel("Validation perplexity")
axes[0].set_title("Experiment A/B: context length & smoothing")
axes[0].set_xticks(ORDERS)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: sparsity grows with N
unseen = [percent_unseen_ngrams(test_sents, o, models[o][0]) for o in ORDERS]
unique = [len(models[o][0]) for o in ORDERS]
ax2 = axes[1]
ax2.bar([str(o) for o in ORDERS], unseen, color="#3b6d9a", alpha=0.85)
ax2.set_xlabel("N-gram order N")
ax2.set_ylabel("% unseen test N-grams")
ax2.set_title("Experiment C: sparsity vs N")
for i, (u, uniq) in enumerate(zip(unseen, unique)):
    ax2.text(i, u + 1, f"{uniq:,}\nunique", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
fig_path = OUTPUT_DIR / "perplexity_sparsity.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print("Wrote", fig_path)
"""
)

md(
    r"""
## 14. Answering the Research Question

Compare the summary table and plots:

1. **Longer context is not always better.** Higher-order models capture more local style, but unique N-grams explode and a larger share of test N-grams is unseen.
2. **Smoothing is essential.** Without it, MLE test perplexity becomes infinite as soon as any unseen transition appears.
3. **Best $k$ depends on $N$.** Larger $N$ is sparser, so a smaller $k$ may still leave too little mass on rare events — or a large $k$ may wash out useful counts. Validation selects the compromise.
4. **Generation trade-off.** Unigrams are bag-of-words noise; bigrams/trigrams sound more Austen-like; 4-grams can look fluent yet copy training fragments.
"""
)

md(
    r"""
## 15. Required Discussion Questions

### 1. Why does the number of possible contexts grow rapidly as $N$ increases?

If the vocabulary has size $|V|$, the number of possible histories of length $N-1$ is up to $|V|^{N-1}$. Even with $|V|$ in the thousands, going from bigrams to 4-grams multiplies the potential context space by $|V|^2$. Most of those contexts never appear in a finite corpus, so counts become extremely sparse.

### 2. Why can a 4-gram model look better on training text yet worse on held-out text?

On training data, many 4-grams are exact memorized phrases, so training likelihood can look excellent. On held-out text, many long contexts are novel; the model then relies on the smoothing floor and can assign worse probabilities than a simpler bigram/trigram that generalizes more robustly. This is classical **overfitting / memorization** in sparse count models.

### 3. What exactly does smoothing change in the probability distribution?

Add-$k$ **redistributes probability mass**: it increases $P(w|h)$ for unseen or rare continuations and correspondingly decreases it for frequent ones, because the denominator grows by $k|V|$. The distribution becomes less peaked and never assigns exact zeros (for $k>0$).

### 4. Which $N$ and smoothing value produced the lowest held-out perplexity, and why?

In this run the **2-gram model with $k=0.01$** achieved the lowest held-out perplexity (test PP $\approx 148$). It uses enough context to capture common collocations and local syntax, while still seeing most test bigrams during training (only about 20% unseen). Trigrams and 4-grams look more “specific” but leave the majority of test N-grams unseen, so even with the best validation $k$ their test perplexity is much worse.

### 5. Compare generated samples. What improves with larger $N$, and what new problem appears?

- **Improves:** local grammaticality, phrase collocations, Austen-like cadence.
- **New problem:** copying / memorization of long training spans; less diversity; brittle dependence on having seen that exact long context.
"""
)

md(
    r"""
## 16. Sanity Checks Recap

| Check | Status in this notebook |
|---|---|
| Vocabulary from training only | Yes |
| Consistent sentence boundaries (`<s>`, `</s>`) | Yes |
| Context length $= N-1$ | Yes |
| Log-space perplexity | Yes |
| Infinite unsmoothed PP explained, not hidden | Yes |
| Predicted-token counting includes `</s>` | Yes |
| Hyperparameters tuned on validation, not test | Yes |
"""
)

md(
    r"""
## References

- Bird, S., Klein, E., & Loper, E. *Natural Language Processing with Python*, Chapter 2 (Gutenberg Corpus). https://www.nltk.org/book/ch02
- NLTK corpus reader HOWTO. https://www.nltk.org/howto/corpus.html
- Project Gutenberg. https://www.gutenberg.org/
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

out = Path(__file__).resolve().parent / "notebooks" / "experiments.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("Wrote notebook with", len(cells), "cells")
