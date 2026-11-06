"""
Train and evaluate the required Project 4 configurations (A–F).

Model selection uses the validation split only.
The official test set is evaluated once for the chosen best model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from load_data import load_splits


# Shared knobs so representation is the main experimental factor
MAX_FEATURES = 150_000
MIN_DF = 2
RANDOM_STATE = 42


def build_config(config_id: str) -> Pipeline:
    """Return a sklearn Pipeline for configuration A–F."""
    configs = {
        "A": {
            "vectorizer": CountVectorizer(
                lowercase=True,
                ngram_range=(1, 1),
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
            ),
            "clf": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "name": "BoW (1,1) + LR",
        },
        "B": {
            "vectorizer": CountVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
            ),
            "clf": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "name": "BoW (1,2) + LR",
        },
        "C": {
            "vectorizer": TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 1),
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
            ),
            "clf": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "name": "TF-IDF (1,1) + LR",
        },
        "D": {
            "vectorizer": TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
            ),
            "clf": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "name": "TF-IDF (1,2) + LR",
        },
        "E": {
            "vectorizer": TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
                sublinear_tf=True,
            ),
            "clf": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "name": "TF-IDF (1,3) + LR",
        },
        "F": {
            "vectorizer": CountVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=MIN_DF,
                max_features=MAX_FEATURES,
            ),
            "clf": MultinomialNB(alpha=1.0),
            "name": "BoW (1,2) + MultinomialNB",
        },
    }
    if config_id not in configs:
        raise ValueError(f"Unknown config {config_id}. Choose from A–F.")

    spec = configs[config_id]
    pipe = Pipeline(
        [
            ("vectorizer", spec["vectorizer"]),
            ("clf", spec["clf"]),
        ]
    )
    pipe.config_name = spec["name"]  # type: ignore[attr-defined]
    return pipe


def evaluate_predictions(y_true, y_pred) -> dict:
    """Binary metrics with positive class = 1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }


def run_all_configs(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Fit each configuration on train and score on validation."""
    config_ids = config_ids or list("ABCDEF")
    rows = []
    for cid in config_ids:
        print(f"\n=== Training config {cid} ===")
        model = build_config(cid)
        model.fit(train["text"], train["label"])
        pred = model.predict(valid["text"])
        metrics = evaluate_predictions(valid["label"], pred)
        n_features = len(model.named_steps["vectorizer"].get_feature_names_out())
        row = {
            "ID": cid,
            "name": model.config_name,  # type: ignore[attr-defined]
            "n_features": n_features,
            "val_accuracy": metrics["accuracy"],
            "val_precision": metrics["precision"],
            "val_recall": metrics["recall"],
            "val_f1": metrics["f1"],
        }
        rows.append(row)
        print(
            f"{cid} ({row['name']}): "
            f"features={n_features}, "
            f"acc={metrics['accuracy']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        )
    return pd.DataFrame(rows)


def top_coefficients(model: Pipeline, k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return strongest positive / negative Logistic Regression N-gram weights."""
    if not hasattr(model.named_steps["clf"], "coef_"):
        raise ValueError("Classifier has no coef_ (need Logistic Regression).")
    features = model.named_steps["vectorizer"].get_feature_names_out()
    weights = model.named_steps["clf"].coef_[0]
    pos_idx = np.argsort(weights)[-k:][::-1]
    neg_idx = np.argsort(weights)[:k]
    pos = pd.DataFrame({"feature": features[pos_idx], "weight": weights[pos_idx]})
    neg = pd.DataFrame({"feature": features[neg_idx], "weight": weights[neg_idx]})
    return pos, neg


def save_confusion_matrix(y_true, y_pred, out_path: Path, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 4 sentiment training")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to aclImdb folder (default: project4_sentiment/data/aclImdb)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train, valid, test = load_splits(data_dir=args.data_dir)
    print(f"Sizes -> train={len(train)}, valid={len(valid)}, test={len(test)}")

    results = run_all_configs(train, valid)
    results_path = out_dir / "model_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nSaved validation results to {results_path}")

    # Select by validation F1 (ties broken by accuracy)
    best_id = results.sort_values(
        ["val_f1", "val_accuracy"], ascending=False
    ).iloc[0]["ID"]
    print(f"\nBest config on validation: {best_id}")

    # Retrain on all official training reviews, evaluate once on official test
    train_full = pd.concat([train, valid], ignore_index=True)
    final_model = build_config(best_id)
    final_model.fit(train_full["text"], train_full["label"])
    test_pred = final_model.predict(test["text"])
    test_metrics = evaluate_predictions(test["label"], test_pred)

    print("\nOfficial test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(classification_report(test["label"], test_pred, digits=4))

    save_confusion_matrix(
        test["label"],
        test_pred,
        out_dir / "confusion_matrix.png",
        title=f"Test Confusion Matrix (config {best_id})",
    )

    summary = {
        "best_config": best_id,
        "best_name": final_model.config_name,  # type: ignore[attr-defined]
        "test_metrics": test_metrics,
        "n_features": int(
            len(final_model.named_steps["vectorizer"].get_feature_names_out())
        ),
    }
    with open(out_dir / "final_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if hasattr(final_model.named_steps["clf"], "coef_"):
        pos, neg = top_coefficients(final_model, k=20)
        pos.to_csv(out_dir / "top_positive_features.csv", index=False)
        neg.to_csv(out_dir / "top_negative_features.csv", index=False)

    print(f"\nArtifacts written under {out_dir}")


if __name__ == "__main__":
    main()
