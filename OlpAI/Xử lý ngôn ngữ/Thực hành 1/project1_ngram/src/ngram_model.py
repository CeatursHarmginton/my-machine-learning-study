"""
N-gram language model utilities for NLP Project 1.

Implements counting, MLE, add-k smoothing, log-probability,
perplexity, next-word prediction, and sampling-based generation.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

History = Tuple[str, ...]
NgramCounts = Counter
ContextCounts = Counter


def normalize_sentence(sent: Sequence[str]) -> List[str]:
    """Lowercase tokens and drop empty strings after stripping."""
    out: List[str] = []
    for tok in sent:
        tok = tok.lower().strip()
        if tok:
            out.append(tok)
    return out


def build_vocabulary(train_sents: Sequence[Sequence[str]], unk_threshold: int = 1) -> set:
    """
    Build vocabulary from training sentences only.

    Words with count <= unk_threshold are treated as rare and excluded;
    <UNK> and </s> are always included.
    """
    train_counts = Counter(w for s in train_sents for w in s)
    vocab = {w for w, c in train_counts.items() if c > unk_threshold}
    vocab.update({"<UNK>", "</s>"})
    return vocab


def apply_unk(sents: Sequence[Sequence[str]], vocab: set) -> List[List[str]]:
    """Map out-of-vocabulary tokens to <UNK>."""
    return [[w if w in vocab else "<UNK>" for w in s] for s in sents]


def ngram_counts(sentences: Sequence[Sequence[str]], order: int) -> Tuple[NgramCounts, ContextCounts]:
    """
    Count N-grams and their contexts.

    Each sentence is padded with (order - 1) start symbols <s>
    and one end symbol </s>.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    ngram: NgramCounts = Counter()
    context: ContextCounts = Counter()

    for sent in sentences:
        padded = ["<s>"] * (order - 1) + list(sent) + ["</s>"]
        for i in range(order - 1, len(padded)):
            hist = tuple(padded[i - order + 1 : i]) if order > 1 else tuple()
            word = padded[i]
            ngram[hist + (word,)] += 1
            context[hist] += 1

    return ngram, context


def mle_probability(
    word: str,
    history: Sequence[str],
    ngram: NgramCounts,
    context: ContextCounts,
) -> float:
    """Maximum-likelihood P(word | history). Returns 0.0 for unseen contexts."""
    history_t = tuple(history)
    denom = context[history_t]
    if denom == 0:
        return 0.0
    return ngram[history_t + (word,)] / denom


def add_k_probability(
    word: str,
    history: Sequence[str],
    ngram: NgramCounts,
    context: ContextCounts,
    vocab_size: int,
    k: float = 0.1,
) -> float:
    """Additive (Lidstone) smoothing: P_k(w|h) = (C(h,w)+k) / (C(h)+k|V|)."""
    history_t = tuple(history)
    numerator = ngram[history_t + (word,)] + k
    denominator = context[history_t] + k * vocab_size
    return numerator / denominator


def sentence_log_probability(
    sent: Sequence[str],
    order: int,
    ngram: NgramCounts,
    context: ContextCounts,
    vocab: set,
    k: float = 0.1,
) -> float:
    """Sum of log P(w_i | h_i) under add-k smoothing."""
    mapped = [w if w in vocab else "<UNK>" for w in sent]
    padded = ["<s>"] * (order - 1) + mapped + ["</s>"]
    total = 0.0
    vocab_size = len(vocab)

    for i in range(order - 1, len(padded)):
        hist = tuple(padded[i - order + 1 : i]) if order > 1 else tuple()
        word = padded[i]
        p = add_k_probability(word, hist, ngram, context, vocab_size, k)
        total += math.log(p)

    return total


def corpus_perplexity(
    sentences: Sequence[Sequence[str]],
    order: int,
    ngram: NgramCounts,
    context: ContextCounts,
    vocab: set,
    k: float = 0.1,
) -> float:
    """
    Held-out perplexity with add-k probabilities in log space.

    PP = exp( -(1/M) * sum log P(w_i | h_i) ), where M counts predicted tokens
    including </s> but not the start markers.
    """
    log_sum = 0.0
    predicted_tokens = 0
    vocab_size = len(vocab)

    for sent in sentences:
        mapped = [w if w in vocab else "<UNK>" for w in sent]
        padded = ["<s>"] * (order - 1) + mapped + ["</s>"]

        for i in range(order - 1, len(padded)):
            hist = tuple(padded[i - order + 1 : i]) if order > 1 else tuple()
            word = padded[i]
            p = add_k_probability(word, hist, ngram, context, vocab_size, k)
            log_sum += math.log(p)
            predicted_tokens += 1

    if predicted_tokens == 0:
        return float("inf")
    return math.exp(-log_sum / predicted_tokens)


def mle_corpus_perplexity(
    sentences: Sequence[Sequence[str]],
    order: int,
    ngram: NgramCounts,
    context: ContextCounts,
    vocab: set,
) -> float:
    """
    Perplexity under unsmoothed MLE.

    Returns infinity if any predicted token has probability 0
    (the classic sparsity failure mode).
    """
    log_sum = 0.0
    predicted_tokens = 0

    for sent in sentences:
        mapped = [w if w in vocab else "<UNK>" for w in sent]
        padded = ["<s>"] * (order - 1) + mapped + ["</s>"]

        for i in range(order - 1, len(padded)):
            hist = tuple(padded[i - order + 1 : i]) if order > 1 else tuple()
            word = padded[i]
            p = mle_probability(word, hist, ngram, context)
            if p <= 0.0:
                return float("inf")
            log_sum += math.log(p)
            predicted_tokens += 1

    if predicted_tokens == 0:
        return float("inf")
    return math.exp(-log_sum / predicted_tokens)


def predict_next(
    history: Sequence[str],
    ngram: NgramCounts,
    context: ContextCounts,
    vocab: set,
    k_smooth: float = 0.1,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return the top-k vocabulary words by add-k probability given history."""
    candidates = []
    for word in vocab:
        if word == "<UNK>":
            continue
        p = add_k_probability(word, tuple(history), ngram, context, len(vocab), k_smooth)
        candidates.append((word, p))
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]


def sample_next(
    history: Sequence[str],
    ngram: NgramCounts,
    context: ContextCounts,
    vocab: set,
    k: float = 0.1,
) -> str:
    """
    Sample one next word from the add-k distribution over the full vocabulary.

    Note: with large |V|, most probability mass can go to unseen words, so
    samples often look noisy. Prefer sample_next_backoff for generation demos.
    """
    words = [w for w in vocab if w != "<UNK>"]
    probs = [add_k_probability(w, history, ngram, context, len(vocab), k) for w in words]
    return random.choices(words, weights=probs, k=1)[0]


def _observed_continuations(
    history: Sequence[str],
    ngram: NgramCounts,
) -> List[Tuple[str, int]]:
    """Return (word, count) pairs observed after this exact history."""
    history_t = tuple(history)
    prefix_len = len(history_t)
    out: List[Tuple[str, int]] = []
    for gram, count in ngram.items():
        if len(gram) == prefix_len + 1 and gram[:prefix_len] == history_t:
            out.append((gram[-1], count))
    return out


def sample_next_backoff(
    history: Sequence[str],
    models: Dict[int, Tuple[NgramCounts, ContextCounts]],
    vocab: set,
    k: float = 0.1,
) -> str:
    """
    Sample from observed continuations, backing off to shorter histories.

    Uses add-k only over the *observed* support of the chosen history.
    Falls back to the unigram model if no shorter history has observations.
    This avoids the near-uniform full-vocabulary distribution that add-k
    creates when |V| is large.
    """
    hist = list(history)
    while True:
        order = len(hist) + 1
        ngram, context = models[order]
        observed = _observed_continuations(hist, ngram)
        # Prefer non-UNK observed words; keep </s> so sentences can end.
        observed = [(w, c) for w, c in observed if w != "<UNK>"]
        if observed:
            words = [w for w, _ in observed]
            # Renormalized add-k over the observed support only.
            probs = [
                add_k_probability(w, hist, ngram, context, len(vocab), k) for w in words
            ]
            return random.choices(words, weights=probs, k=1)[0]
        if not hist:
            break
        hist = hist[1:]  # drop the oldest token and back off

    # Absolute fallback: smoothed unigram over the full vocabulary
    uni_ng, uni_ctx = models[1]
    return sample_next(tuple(), uni_ng, uni_ctx, vocab, k)


def generate(
    order: int,
    ngram: NgramCounts,
    context: ContextCounts,
    vocab: set,
    max_tokens: int = 100,
    k: float = 0.1,
    seed: Optional[int] = 7,
    models: Optional[Dict[int, Tuple[NgramCounts, ContextCounts]]] = None,
    use_backoff: bool = True,
) -> str:
    """
    Generate a sentence by sampling until </s> or max_tokens.

    If models is provided and use_backoff is True (default), sample with
    observed-continuation backoff. Otherwise use full-vocabulary add-k.
    """
    if seed is not None:
        random.seed(seed)

    history: List[str] = ["<s>"] * (order - 1)
    output: List[str] = []

    for _ in range(max_tokens):
        if use_backoff and models is not None:
            word = sample_next_backoff(tuple(history), models, vocab, k)
        else:
            word = sample_next(tuple(history), ngram, context, vocab, k)
        if word == "</s>":
            break
        output.append(word)
        if order > 1:
            history = (history + [word])[-(order - 1) :]

    return " ".join(output)


def interpolate_probability(
    word: str,
    history: Sequence[str],
    models: Dict[int, Tuple[NgramCounts, ContextCounts]],
    lambdas: Sequence[float],
    vocab_size: int,
    k: float = 0.1,
) -> float:
    """
    Linear interpolation of add-k models of different orders.

    models maps order -> (ngram_counts, context_counts).
    lambdas[i] corresponds to order i+1 (unigram, bigram, ...).
    """
    if abs(sum(lambdas) - 1.0) > 1e-6:
        raise ValueError("lambdas must sum to 1")

    total = 0.0
    for order, lam in enumerate(lambdas, start=1):
        ngram, context = models[order]
        # Use the last (order-1) tokens as history for that order.
        hist = tuple(history[-(order - 1) :]) if order > 1 else tuple()
        total += lam * add_k_probability(word, hist, ngram, context, vocab_size, k)
    return total


def interpolated_corpus_perplexity(
    sentences: Sequence[Sequence[str]],
    max_order: int,
    models: Dict[int, Tuple[NgramCounts, ContextCounts]],
    lambdas: Sequence[float],
    vocab: set,
    k: float = 0.1,
) -> float:
    """Perplexity using interpolated probabilities with context length max_order-1."""
    log_sum = 0.0
    predicted_tokens = 0
    vocab_size = len(vocab)

    for sent in sentences:
        mapped = [w if w in vocab else "<UNK>" for w in sent]
        padded = ["<s>"] * (max_order - 1) + mapped + ["</s>"]

        for i in range(max_order - 1, len(padded)):
            hist = tuple(padded[i - max_order + 1 : i])
            word = padded[i]
            p = interpolate_probability(word, hist, models, lambdas, vocab_size, k)
            log_sum += math.log(p)
            predicted_tokens += 1

    return math.exp(-log_sum / predicted_tokens)


def percent_unseen_ngrams(
    test_sents: Sequence[Sequence[str]],
    order: int,
    train_ngrams: NgramCounts,
) -> float:
    """Percentage of test N-grams that never appeared in training."""
    total = 0
    unseen = 0
    for sent in test_sents:
        padded = ["<s>"] * (order - 1) + list(sent) + ["</s>"]
        for i in range(order - 1, len(padded)):
            gram = tuple(padded[i - order + 1 : i + 1])
            total += 1
            if train_ngrams[gram] == 0:
                unseen += 1
    return 100.0 * unseen / total if total else 0.0


def observed_continuation_prob_sum(
    history: Sequence[str],
    ngram: NgramCounts,
    context: ContextCounts,
) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Sum MLE probabilities over observed continuations of a history.

    Useful as a sanity check that probabilities sum to ~1.0 for seen contexts.
    """
    history_t = tuple(history)
    denom = context[history_t]
    if denom == 0:
        return 0.0, []

    continuations: List[Tuple[str, float]] = []
    total = 0.0
    # Scan ngrams that share this exact history prefix.
    prefix_len = len(history_t)
    for gram, count in ngram.items():
        if gram[:prefix_len] == history_t and len(gram) == prefix_len + 1:
            word = gram[-1]
            p = count / denom
            continuations.append((word, p))
            total += p

    continuations.sort(key=lambda x: x[1], reverse=True)
    return total, continuations
