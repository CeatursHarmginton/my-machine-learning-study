from __future__ import annotations

import sys
import shutil
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mnist_fashion_research.config import DATASET_SPECS


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def common_setup(dataset_name: str):
    return code(
        f"""
        from pathlib import Path
        import json
        import math
        import random
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import HTML, display

        ROOT = Path.cwd().resolve()
        while ROOT != ROOT.parent and not (ROOT / "src").exists():
            ROOT = ROOT.parent
        sys.path.insert(0, str(ROOT / "src"))

        DATASET_NAME = "{dataset_name}"
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)

        from mnist_fashion_research.config import get_dataset_spec
        from mnist_fashion_research.data import (
            dataset_card,
            load_numpy_split,
            load_tensor_split,
            make_dataloaders,
        )
        from mnist_fashion_research.visualization import (
            animate_augmentation,
            animate_cumulative_mean,
            animate_samples,
            animate_training_curve,
            plot_confusion_matrix,
            plot_image_grid,
        )

        spec = get_dataset_spec(DATASET_NAME)
        class_names = list(spec.class_names)
        print(f"Project root: {{ROOT}}")
        print(f"Dataset: {{spec.display_name}} with {{len(class_names)}} classes")
        """
    )


def data_load_cell():
    return code(
        """
        X_train, y_train = load_tensor_split(DATASET_NAME, "train", root=ROOT)
        X_test, y_test = load_tensor_split(DATASET_NAME, "test", root=ROOT)

        train_np = X_train.numpy()
        test_np = X_test.numpy()
        y_train_np = y_train.numpy()
        y_test_np = y_test.numpy()

        print("Train:", tuple(X_train.shape), tuple(y_train.shape), X_train.dtype)
        print("Test:", tuple(X_test.shape), tuple(y_test.shape), X_test.dtype)
        """
    )


def problem_framing_cells(dataset_name: str):
    return [
        md(
            f"""
            # 00 - Problem Framing: {dataset_name}

            A professional ML project starts before training a model. In this
            notebook we define the task, success criteria, experiment scope, and
            reproducibility rules for {dataset_name}.

            The central task is single-label image classification. Each input is
            a 28 by 28 grayscale image, and the model must predict one of 10
            classes. We will compare simple baselines against LeNet and a compact
            AlexNet-style CNN under a consistent pipeline.
            """
        ),
        common_setup(dataset_name),
        md(
            """
            ## Research Questions

            We will keep the questions concrete:

            - How much performance do simple models achieve before using CNNs?
            - Does LeNet provide a meaningful improvement over linear and MLP
              baselines?
            - Does a compact AlexNet-style model improve accuracy, robustness, or
              only add complexity?
            - Which classes and visual patterns create the most errors?
            - Which preprocessing and augmentation choices are justified by the
              data rather than copied from a tutorial?
            """
        ),
        code(
            """
            card = dataset_card(DATASET_NAME)
            pd.DataFrame(
                [
                    {"property": key, "value": value}
                    for key, value in card.items()
                    if key not in {"class_names"}
                ]
            )
            """
        ),
        md(
            """
            ## Success Criteria

            Accuracy alone is not enough. A research-quality comparison should
            report overall accuracy, macro F1, per-class accuracy, confusion
            matrices, training curves, and error examples. We also save seeds,
            dataset splits, model checkpoints, and histories so results can be
            reproduced.
            """
        ),
        code(
            """
            criteria = pd.DataFrame(
                [
                    ["Primary metric", "Test accuracy"],
                    ["Secondary metrics", "Macro F1, per-class accuracy, confusion matrix"],
                    ["Reliability checks", "Train/validation curves, calibration, robustness"],
                    ["Models", "Majority, nearest centroid, logistic regression, MLP, LeNet, AlexNetMini"],
                    ["Reproducibility", "Fixed seed, saved configs, checkpoints, versioned outputs"],
                ],
                columns=["item", "decision"],
            )
            criteria
            """
        ),
        md(
            """
            ## Workflow Map

            The project uses an iterative research loop. EDA informs
            preprocessing. Baselines set expectations. CNNs are trained only
            after the baseline evidence is available. Evaluation sends us back
            to data inspection when errors reveal a pattern.
            """
        ),
        code(
            """
            stages = [
                "Frame",
                "EDA",
                "Preprocess",
                "Baselines",
                "LeNet/AlexNet",
                "Train",
                "Evaluate",
                "Interpret",
                "Report",
            ]
            fig, ax = plt.subplots(figsize=(11, 2.5))
            ax.set_xlim(-0.5, len(stages) - 0.5)
            ax.set_ylim(0, 1)
            for i, stage in enumerate(stages):
                ax.scatter(i, 0.5, s=900, color="#4C78A8")
                ax.text(i, 0.5, stage, color="white", ha="center", va="center", fontsize=9)
                if i < len(stages) - 1:
                    ax.annotate("", xy=(i + 0.72, 0.5), xytext=(i + 0.28, 0.5),
                                arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
            ax.axis("off")
            plt.show()
            """
        ),
        code(
            """
            from matplotlib import animation

            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.set_xlim(-0.5, len(stages) - 0.5)
            ax.set_ylim(0, 1)
            points = ax.scatter(range(len(stages)), [0.5] * len(stages), s=650, color="#BBBBBB")
            for i, stage in enumerate(stages):
                ax.text(i, 0.5, stage, ha="center", va="center", fontsize=8)
            ax.axis("off")

            def update(frame):
                colors = ["#BBBBBB"] * len(stages)
                colors[frame] = "#F58518"
                points.set_color(colors)
                ax.set_title(f"Current research focus: {stages[frame]}")
                return (points,)

            anim = animation.FuncAnimation(fig, update, frames=len(stages), interval=700, blit=False)
            plt.close(fig)
            HTML(anim.to_jshtml())
            """
        ),
    ]


def eda_cells(dataset_name: str):
    return [
        md(
            f"""
            # 01 - Exploratory Data Analysis: {dataset_name}

            EDA is the evidence-gathering stage. We check whether the data are
            valid, balanced, visually sensible, statistically structured, and
            appropriate for the models we plan to train.
            """
        ),
        common_setup(dataset_name),
        data_load_cell(),
        md(
            """
            ## 1. Dataset Integrity

            First we verify the split sizes, tensor shapes, dtype, pixel range,
            and label range. These checks catch many silent bugs: wrong file,
            wrong split, accidental normalization before export, or labels that
            do not match the declared classes.
            """
        ),
        code(
            """
            integrity = pd.DataFrame(
                [
                    ["train images", tuple(X_train.shape)],
                    ["test images", tuple(X_test.shape)],
                    ["train dtype", str(X_train.dtype)],
                    ["test dtype", str(X_test.dtype)],
                    ["pixel min/max", f"{int(X_train.min())} / {int(X_train.max())}"],
                    ["label min/max", f"{int(y_train.min())} / {int(y_train.max())}"],
                    ["classes observed", sorted(np.unique(y_train_np).tolist())],
                ],
                columns=["check", "value"],
            )
            display(integrity)

            assert X_train.ndim == 3 and X_train.shape[1:] == (28, 28)
            assert X_test.ndim == 3 and X_test.shape[1:] == (28, 28)
            assert X_train.min() >= 0 and X_train.max() <= 255
            assert set(np.unique(y_train_np)) == set(range(10))
            """
        ),
        md(
            """
            ## 2. Class Distribution

            Class balance affects the choice of metric and loss. MNIST-like
            datasets are close to balanced, but a professional report should
            show exact counts instead of assuming balance.
            """
        ),
        code(
            """
            counts = pd.Series(y_train_np).value_counts().sort_index()
            distribution = pd.DataFrame(
                {
                    "label_id": counts.index,
                    "label_name": [class_names[i] for i in counts.index],
                    "count": counts.values,
                    "percentage": counts.values / counts.values.sum() * 100,
                }
            )
            display(distribution)

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(distribution["label_name"], distribution["count"], color="#4C78A8")
            ax.set_title(f"{DATASET_NAME}: training class distribution")
            ax.set_ylabel("images")
            ax.tick_params(axis="x", rotation=45)
            plt.show()
            """
        ),
        md(
            """
            ## 3. Visual Inspection

            A sample grid is not enough for a full EDA, but it is still a
            necessary sanity check. We want to see whether labels, alignment,
            contrast, and visual variability match the dataset description.
            """
        ),
        code(
            """
            rng = np.random.default_rng(SEED)
            sample_indices = []
            for label in range(10):
                candidates = np.where(y_train_np == label)[0]
                sample_indices.extend(rng.choice(candidates, size=8, replace=False))
            plot_image_grid(
                X_train[sample_indices],
                y_train[sample_indices],
                class_names=class_names,
                ncols=8,
                title=f"{DATASET_NAME}: random samples per class",
            )
            plt.show()
            """
        ),
        md(
            """
            ## 4. Pixel Intensity and Sparsity

            Pixel histograms reveal whether most of the image is background,
            whether values are already normalized, and whether a binarization or
            normalization decision might be reasonable.
            """
        ),
        code(
            """
            flat_pixels = train_np.reshape(-1)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(flat_pixels, bins=50, color="#59A14F", alpha=0.85)
            ax.set_title(f"{DATASET_NAME}: global pixel intensity distribution")
            ax.set_xlabel("pixel value")
            ax.set_ylabel("frequency")
            plt.show()

            global_stats = pd.DataFrame(
                [
                    ["mean", flat_pixels.mean()],
                    ["std", flat_pixels.std()],
                    ["median", np.median(flat_pixels)],
                    ["active pixel ratio (pixel > 0)", np.mean(flat_pixels > 0)],
                    ["high intensity ratio (pixel > 127)", np.mean(flat_pixels > 127)],
                ],
                columns=["statistic", "value"],
            )
            global_stats
            """
        ),
        md(
            """
            ## 5. Per-Class Statistics

            Different classes occupy different amounts of space. A digit "1" or
            a sandal usually has fewer active pixels than a digit "8" or a coat.
            These differences matter because models can accidentally use area or
            thickness as shortcuts.
            """
        ),
        code(
            """
            yy, xx = np.indices((28, 28))
            rows = []
            for label in range(10):
                cls = train_np[y_train_np == label].astype(np.float32)
                mass = cls.sum(axis=(1, 2)) + 1e-6
                center_y = (cls * yy).sum(axis=(1, 2)) / mass
                center_x = (cls * xx).sum(axis=(1, 2)) / mass
                rows.append(
                    {
                        "label_id": label,
                        "label_name": class_names[label],
                        "mean_intensity": cls.mean(),
                        "std_intensity": cls.std(),
                        "active_pixel_ratio": (cls > 0).mean(),
                        "mean_center_x": center_x.mean(),
                        "mean_center_y": center_y.mean(),
                    }
                )
            per_class_stats = pd.DataFrame(rows)
            display(per_class_stats)
            """
        ),
        md(
            """
            ## 6. Mean Images and Variance Maps

            The mean image is a class prototype. The standard deviation map
            shows where examples vary most. Together they tell us where a model
            should find stable evidence and where it may face ambiguity.
            """
        ),
        code(
            """
            fig, axes = plt.subplots(4, 5, figsize=(12, 8))
            for label in range(10):
                cls = train_np[y_train_np == label].astype(np.float32)
                mean_img = cls.mean(axis=0)
                std_img = cls.std(axis=0)
                axes[0 if label < 5 else 2, label % 5].imshow(mean_img, cmap="gray")
                axes[0 if label < 5 else 2, label % 5].set_title(f"Mean: {class_names[label]}")
                axes[1 if label < 5 else 3, label % 5].imshow(std_img, cmap="magma")
                axes[1 if label < 5 else 3, label % 5].set_title(f"Std: {class_names[label]}")
            for ax in axes.ravel():
                ax.axis("off")
            fig.suptitle(f"{DATASET_NAME}: per-class mean and standard deviation maps", y=1.02)
            fig.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## 7. Duplicate and Leakage Checks

            Exact duplicates and train/test overlap can make evaluation look
            better than it really is. We use image hashes to check for exact
            overlap. Near-duplicates are more complex, but exact hashing is a
            strong first audit.
            """
        ),
        code(
            """
            import hashlib

            def image_hashes(array):
                return [hashlib.sha1(image.tobytes()).hexdigest() for image in array]

            train_hashes = image_hashes(train_np)
            test_hashes = image_hashes(test_np)
            duplicate_train = len(train_hashes) - len(set(train_hashes))
            leakage = len(set(train_hashes).intersection(test_hashes))

            pd.DataFrame(
                [
                    ["duplicate training images", duplicate_train],
                    ["exact train/test overlap", leakage],
                ],
                columns=["audit", "count"],
            )
            """
        ),
        md(
            """
            ## 8. PCA Structure

            PCA answers two questions. First, how compressible are the images by
            a linear method? Second, do the first few directions already separate
            classes? Strong nonlinear models are still useful, but PCA gives a
            transparent baseline view of the geometry.
            """
        ),
        code(
            """
            from sklearn.decomposition import PCA

            pca_sample_size = min(10000, len(train_np))
            pca_indices = rng.choice(len(train_np), size=pca_sample_size, replace=False)
            X_pca = train_np[pca_indices].reshape(pca_sample_size, -1).astype("float32") / 255.0
            y_pca = y_train_np[pca_indices]

            pca = PCA(n_components=80, random_state=SEED)
            X_pca_2d = pca.fit_transform(X_pca)
            cumulative = np.cumsum(pca.explained_variance_ratio_)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            axes[0].plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", ms=3)
            axes[0].set_title("PCA cumulative explained variance")
            axes[0].set_xlabel("components")
            axes[0].set_ylabel("cumulative variance")
            scatter = axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_pca, s=8, cmap="tab10", alpha=0.7)
            axes[1].set_title("First two PCA components")
            axes[1].set_xlabel("PC1")
            axes[1].set_ylabel("PC2")
            fig.colorbar(scatter, ax=axes[1], ticks=range(10))
            plt.show()

            thresholds = [0.80, 0.90, 0.95]
            pd.DataFrame(
                {
                    "variance_target": thresholds,
                    "components_needed": [int(np.searchsorted(cumulative, t) + 1) for t in thresholds],
                }
            )
            """
        ),
        md(
            """
            ## 9. Prototype Similarity

            The class mean images can be compared as vectors. Similar prototypes
            often predict likely confusion pairs, such as visually similar
            digits or similar clothing categories.
            """
        ),
        code(
            """
            from sklearn.metrics.pairwise import cosine_similarity

            prototypes = np.stack(
                [train_np[y_train_np == label].mean(axis=0).reshape(-1) for label in range(10)]
            )
            similarity = cosine_similarity(prototypes)
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(similarity, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(10), class_names, rotation=45, ha="right")
            ax.set_yticks(range(10), class_names)
            ax.set_title("Cosine similarity between class prototypes")
            fig.colorbar(im, ax=ax)
            plt.show()
            """
        ),
        md(
            """
            ## 10. Outlier Scan

            Outliers can be unusual writing styles, ambiguous clothing, label
            noise, or low-quality images. We detect unusual samples in a PCA
            feature space and inspect them visually.
            """
        ),
        code(
            """
            from sklearn.ensemble import IsolationForest

            pca_features = PCA(n_components=30, random_state=SEED).fit_transform(X_pca)
            detector = IsolationForest(contamination=0.02, random_state=SEED)
            detector.fit(pca_features)
            outlier_scores = -detector.score_samples(pca_features)
            top_local = np.argsort(outlier_scores)[-25:][::-1]
            top_indices = pca_indices[top_local]

            plot_image_grid(
                X_train[top_indices],
                y_train[top_indices],
                class_names=class_names,
                ncols=5,
                title=f"{DATASET_NAME}: candidate outliers from PCA + IsolationForest",
            )
            plt.show()
            """
        ),
        md(
            """
            ## 11. EDA Animations

            The following animations show concepts that are easier to understand
            over time: random sample variation, the formation of a class
            prototype, and how thresholding changes the visible foreground.
            """
        ),
        code(
            """
            ordered = []
            for label in range(10):
                ordered.extend(np.where(y_train_np == label)[0][:3])
            display(animate_samples(X_train[ordered], y_train[ordered], class_names, n_frames=len(ordered)))
            """
        ),
        code(
            """
            focus_label = 0
            focus_indices = np.where(y_train_np == focus_label)[0][:60]
            display(animate_cumulative_mean(X_train[focus_indices], class_names[focus_label], max_frames=60))
            """
        ),
        code(
            """
            from matplotlib import animation

            example = train_np[sample_indices[0]]
            thresholds = np.linspace(0, 255, 32)
            fig, ax = plt.subplots(figsize=(3, 3))
            display_img = ax.imshow(example > thresholds[0], cmap="gray")
            ax.axis("off")

            def update(frame):
                threshold = thresholds[frame]
                display_img.set_data(example > threshold)
                ax.set_title(f"Foreground threshold > {threshold:.0f}")
                return (display_img,)

            anim = animation.FuncAnimation(fig, update, frames=len(thresholds), interval=180, blit=False)
            plt.close(fig)
            HTML(anim.to_jshtml())
            """
        ),
    ]


def preprocessing_cells(dataset_name: str):
    return [
        md(
            f"""
            # 02 - Preprocessing and Augmentation: {dataset_name}

            Preprocessing and augmentation solve different problems.
            Preprocessing makes inputs numerically stable and consistent.
            Augmentation teaches the model that small visual changes should not
            change the class.
            """
        ),
        common_setup(dataset_name),
        data_load_cell(),
        md(
            """
            ## 1. Compute Dataset Statistics

            Normalization should be justified from the data. We compute the
            train-set mean and standard deviation on pixels scaled to [0, 1].
            The test set is never used to choose preprocessing constants.
            """
        ),
        code(
            """
            X_float = X_train.float() / 255.0
            computed_mean = float(X_float.mean())
            computed_std = float(X_float.std())
            pd.DataFrame(
                [
                    ["computed train mean", computed_mean],
                    ["computed train std", computed_std],
                    ["canonical mean in config", spec.canonical_mean],
                    ["canonical std in config", spec.canonical_std],
                ],
                columns=["statistic", "value"],
            )
            """
        ),
        code(
            """
            normalized = (X_float - computed_mean) / computed_std
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(X_float.flatten().numpy(), bins=50, color="#4C78A8")
            axes[0].set_title("Before normalization")
            axes[1].hist(normalized.flatten().numpy(), bins=50, color="#F58518")
            axes[1].set_title("After normalization")
            plt.show()
            """
        ),
        md(
            """
            ## 2. Train/Validation/Test Policy

            The original test set stays untouched until final evaluation. We
            split the original training set into train and validation subsets
            with stratification so every class keeps similar proportions.
            """
        ),
        code(
            """
            from sklearn.model_selection import StratifiedShuffleSplit

            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
            train_idx, val_idx = next(splitter.split(np.zeros(len(y_train_np)), y_train_np))
            split_table = pd.DataFrame(
                {
                    "split": ["train", "validation", "test"],
                    "images": [len(train_idx), len(val_idx), len(y_test_np)],
                }
            )
            display(split_table)

            split_counts = []
            for split_name, labels in [
                ("train", y_train_np[train_idx]),
                ("validation", y_train_np[val_idx]),
                ("test", y_test_np),
            ]:
                counts = pd.Series(labels).value_counts(normalize=True).sort_index()
                for label_id, pct in counts.items():
                    split_counts.append(
                        {"split": split_name, "label": class_names[label_id], "percentage": pct * 100}
                    )
            split_counts = pd.DataFrame(split_counts)
            split_counts.pivot(index="label", columns="split", values="percentage").round(2)
            """
        ),
        md(
            """
            ## 3. Mild Augmentation Policy

            For 28 by 28 grayscale images, heavy augmentation can change the
            label or erase important structure. We use mild rotation and small
            translations because they match realistic variation in handwriting
            and clothing alignment.
            """
        ),
        code(
            """
            from PIL import Image
            from torchvision import transforms

            augmentation = transforms.Compose(
                [
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
                    transforms.ToTensor(),
                ]
            )
            example_index = int(np.where(y_train_np == 0)[0][0])
            pil_example = Image.fromarray(X_train[example_index].numpy(), mode="L")

            augmented_frames = [augmentation(pil_example).squeeze().numpy() for _ in range(16)]
            plot_image_grid(augmented_frames, labels=[y_train_np[example_index]] * 16,
                            class_names=class_names, ncols=8,
                            title="Mild augmentation samples")
            plt.show()
            """
        ),
        code(
            """
            display(animate_augmentation(pil_example, augmentation, n_frames=24))
            """
        ),
        md(
            """
            ## 4. DataLoader Contract

            The model expects tensors shaped as `(batch, channels, height,
            width)`. The labels remain integer class IDs because
            `CrossEntropyLoss` expects class indices, not one-hot labels.
            """
        ),
        code(
            """
            loaders = make_dataloaders(
                DATASET_NAME,
                batch_size=128,
                val_fraction=0.1,
                seed=SEED,
                augment=True,
                root=ROOT,
            )
            batch_images, batch_labels = next(iter(loaders["train"]))
            print("Batch images:", tuple(batch_images.shape), batch_images.dtype)
            print("Batch labels:", tuple(batch_labels.shape), batch_labels.dtype)
            print("Batch value range after normalization:", float(batch_images.min()), float(batch_images.max()))
            """
        ),
    ]


def baseline_cells(dataset_name: str):
    return [
        md(
            f"""
            # 03 - Baseline Models: {dataset_name}

            Baselines tell us what performance is possible without deep CNNs.
            A CNN result is meaningful only when it beats simple, well-run
            alternatives under the same data split.
            """
        ),
        common_setup(dataset_name),
        md(
            """
            ## 1. Prepare Flattened Inputs

            Classical ML baselines use flattened 784-dimensional vectors. We
            scale pixels to [0, 1] and use a subset by default so the notebook
            runs quickly on a classroom laptop. Increase `TRAIN_LIMIT` for a
            full experiment.
            """
        ),
        code(
            """
            X_train_np, y_train_np = load_numpy_split(DATASET_NAME, "train", root=ROOT)
            X_test_np, y_test_np = load_numpy_split(DATASET_NAME, "test", root=ROOT)

            rng = np.random.default_rng(SEED)
            TRAIN_LIMIT = min(12000, len(y_train_np))
            TEST_LIMIT = min(3000, len(y_test_np))
            train_indices = rng.choice(len(y_train_np), TRAIN_LIMIT, replace=False)
            test_indices = rng.choice(len(y_test_np), TEST_LIMIT, replace=False)

            Xb_train = X_train_np[train_indices].reshape(TRAIN_LIMIT, -1).astype("float32") / 255.0
            yb_train = y_train_np[train_indices]
            Xb_test = X_test_np[test_indices].reshape(TEST_LIMIT, -1).astype("float32") / 255.0
            yb_test = y_test_np[test_indices]
            print(Xb_train.shape, Xb_test.shape)
            """
        ),
        md(
            """
            ## 2. Fit Baselines

            We compare a dummy classifier, nearest centroid, logistic
            regression, and a small MLP. These models are intentionally simpler
            than LeNet and AlexNetMini.
            """
        ),
        code(
            """
            from sklearn.dummy import DummyClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.neighbors import NearestCentroid
            from sklearn.neural_network import MLPClassifier

            estimators = {
                "majority": DummyClassifier(strategy="most_frequent"),
                "nearest_centroid": NearestCentroid(),
                "logistic_regression": LogisticRegression(max_iter=150, n_jobs=-1, random_state=SEED),
                "mlp": MLPClassifier(hidden_layer_sizes=(128,), max_iter=30, random_state=SEED),
            }

            baseline_rows = []
            fitted = {}
            for name, estimator in estimators.items():
                estimator.fit(Xb_train, yb_train)
                predictions = estimator.predict(Xb_test)
                fitted[name] = estimator
                baseline_rows.append(
                    {
                        "model": name,
                        "accuracy": accuracy_score(yb_test, predictions),
                        "macro_f1": f1_score(yb_test, predictions, average="macro"),
                    }
                )
            baseline_results = pd.DataFrame(baseline_rows).sort_values("accuracy", ascending=False)
            baseline_results
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(baseline_results["model"], baseline_results["accuracy"], color="#4C78A8")
            ax.set_ylim(0, 1)
            ax.set_ylabel("accuracy")
            ax.set_title(f"{DATASET_NAME}: baseline accuracy comparison")
            ax.tick_params(axis="x", rotation=30)
            plt.show()
            """
        ),
        md(
            """
            ## 3. Confusion Matrix for the Strongest Baseline

            A confusion matrix turns one accuracy number into an error pattern.
            We inspect the strongest baseline because it sets the benchmark the
            CNNs must beat.
            """
        ),
        code(
            """
            from sklearn.metrics import confusion_matrix

            best_name = baseline_results.iloc[0]["model"]
            best_model = fitted[best_name]
            best_predictions = best_model.predict(Xb_test)
            cm = confusion_matrix(yb_test, best_predictions)
            plot_confusion_matrix(cm, class_names, title=f"{best_name}: confusion matrix")
            plt.show()
            """
        ),
        md(
            """
            ## 4. Baseline Results Animation

            This animation reveals the comparison one model at a time. It is a
            simple way to teach that a model comparison is cumulative evidence,
            not a single final number.
            """
        ),
        code(
            """
            from matplotlib import animation

            ordered = baseline_results.sort_values("accuracy")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, len(ordered) - 0.5)
            bars = ax.barh(ordered["model"], [0] * len(ordered), color="#F58518")
            ax.set_xlabel("accuracy")

            def update(frame):
                for i, bar in enumerate(bars):
                    value = ordered.iloc[i]["accuracy"] if i <= frame else 0
                    bar.set_width(value)
                ax.set_title(f"Baseline comparison: {ordered.iloc[frame]['model']}")
                return bars

            anim = animation.FuncAnimation(fig, update, frames=len(ordered), interval=700, blit=False)
            plt.close(fig)
            HTML(anim.to_jshtml())
            """
        ),
    ]


def lenet_alexnet_cells(dataset_name: str):
    return [
        md(
            f"""
            # 04 - LeNet and AlexNet-Style CNNs: {dataset_name}

            LeNet is a historically important small CNN for digit recognition.
            AlexNet was designed for much larger RGB images, so here we use a
            compact AlexNet-style network adapted to 28 by 28 grayscale inputs.
            """
        ),
        common_setup(dataset_name),
        data_load_cell(),
        md(
            """
            ## 1. Model Definitions and Parameter Counts

            Before training, inspect model capacity. More parameters are not
            automatically better, especially on small images where the task may
            already be close to solved by smaller models.
            """
        ),
        code(
            """
            from mnist_fashion_research.models import (
                AlexNetMini,
                LeNet5,
                MLPBaseline,
                SimpleCNN,
                count_parameters,
                layer_shape_table,
            )

            models = {
                "MLPBaseline": MLPBaseline(),
                "SimpleCNN": SimpleCNN(),
                "LeNet5": LeNet5(),
                "AlexNetMini": AlexNetMini(),
            }
            pd.DataFrame(
                [{"model": name, "trainable_parameters": count_parameters(model)}
                 for name, model in models.items()]
            ).sort_values("trainable_parameters")
            """
        ),
        code(
            """
            pd.DataFrame(layer_shape_table(LeNet5()))
            """
        ),
        code(
            """
            pd.DataFrame(layer_shape_table(AlexNetMini()))
            """
        ),
        md(
            """
            ## 2. Convolution as a Sliding Window

            The animation below applies a small edge-detection kernel across one
            image. Real CNN filters are learned from data, but this demonstrates
            the local receptive field idea that makes CNNs different from MLPs.
            """
        ),
        code(
            """
            from matplotlib import animation

            image = X_train[0].float().numpy() / 255.0
            kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
            output = np.zeros((26, 26), dtype=np.float32)
            positions = [(r, c) for r in range(26) for c in range(26)]
            positions = positions[::18]

            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            axes[0].imshow(image, cmap="gray")
            rect = plt.Rectangle((0, 0), 3, 3, fill=False, edgecolor="#F58518", linewidth=2)
            axes[0].add_patch(rect)
            conv_display = axes[1].imshow(output, cmap="magma", vmin=-3, vmax=3)
            axes[0].set_title("Input")
            axes[1].set_title("Convolution response")
            for ax in axes:
                ax.axis("off")

            def update(frame):
                r, c = positions[frame]
                patch = image[r:r + 3, c:c + 3]
                output[r, c] = float((patch * kernel).sum())
                rect.set_xy((c, r))
                conv_display.set_data(output)
                return rect, conv_display

            anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=80, blit=False)
            plt.close(fig)
            HTML(anim.to_jshtml())
            """
        ),
        md(
            """
            ## 3. First-Layer Feature Maps

            These activations are from an untrained model, so they are not yet
            meaningful features. The purpose is to inspect tensor shapes and
            learn how feature maps can be visualized after training.
            """
        ),
        code(
            """
            model = LeNet5()
            model.eval()
            with torch.no_grad():
                example = (X_train[0].float() / 255.0).unsqueeze(0).unsqueeze(0)
                activation = model.features[0](example).squeeze(0)

            fig, axes = plt.subplots(1, activation.shape[0], figsize=(12, 2))
            for i, ax in enumerate(axes):
                ax.imshow(activation[i].numpy(), cmap="magma")
                ax.set_title(f"map {i}")
                ax.axis("off")
            plt.show()
            """
        ),
    ]


def training_cells(dataset_name: str):
    return [
        md(
            f"""
            # 05 - Training Pipeline: {dataset_name}

            A training pipeline is more than calling `fit`. It controls seeds,
            splits, augmentation, optimization, checkpointing, and logs so the
            experiment can be repeated and compared fairly.
            """
        ),
        common_setup(dataset_name),
        md(
            """
            ## 1. Experiment Configuration

            The defaults below are intentionally classroom-friendly. Increase
            `EPOCHS` and remove the batch limits for a full research run.
            """
        ),
        code(
            """
            from mnist_fashion_research.models import AlexNetMini, LeNet5
            from mnist_fashion_research.training import fit, save_history
            from mnist_fashion_research.utils import ensure_dir, set_seed

            MODEL_NAME = "lenet"  # Choose "lenet" or "alexnet_mini".
            EPOCHS = 3
            MAX_TRAIN_BATCHES = 120
            MAX_EVAL_BATCHES = 40
            BATCH_SIZE = 128
            LR = 1e-3

            model = LeNet5() if MODEL_NAME == "lenet" else AlexNetMini()
            output_dir = ROOT / "outputs" / "models" / spec.slug
            history_path = ROOT / "outputs" / "reports" / f"{spec.slug}_{MODEL_NAME}_history.json"
            checkpoint_path = output_dir / f"{MODEL_NAME}.pt"
            ensure_dir(output_dir)
            """
        ),
        md(
            """
            ## 2. Build DataLoaders

            Training uses augmentation. Validation and test data do not use
            random augmentation, because evaluation should be stable.
            """
        ),
        code(
            """
            loaders = make_dataloaders(
                DATASET_NAME,
                batch_size=BATCH_SIZE,
                val_fraction=0.1,
                seed=SEED,
                augment=True,
                root=ROOT,
            )
            images, labels = next(iter(loaders["train"]))
            print(tuple(images.shape), tuple(labels.shape))
            """
        ),
        md(
            """
            ## 3. Train and Save the Best Checkpoint

            The checkpoint stores the best validation model. The history stores
            train/validation loss and accuracy for later evaluation notebooks.
            """
        ),
        code(
            """
            set_seed(SEED)
            history = fit(
                model,
                loaders,
                epochs=EPOCHS,
                lr=LR,
                seed=SEED,
                checkpoint_path=checkpoint_path,
                max_train_batches=MAX_TRAIN_BATCHES,
                max_eval_batches=MAX_EVAL_BATCHES,
            )
            save_history(history, history_path)
            pd.DataFrame(history)
            """
        ),
        md(
            """
            ## 4. Training Curves

            Curves are diagnostic tools. If training accuracy rises while
            validation accuracy stalls, the model may be overfitting. If both
            stay low, the model may be underfitting or the optimization setup
            may need attention.
            """
        ),
        code(
            """
            history_df = pd.DataFrame(history)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
            axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="validation")
            axes[0].set_title("Loss")
            axes[0].legend()
            axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="train")
            axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="validation")
            axes[1].set_title("Accuracy")
            axes[1].set_ylim(0, 1)
            axes[1].legend()
            plt.show()
            """
        ),
        code(
            """
            display(animate_training_curve(history))
            """
        ),
    ]


def evaluation_cells(dataset_name: str):
    return [
        md(
            f"""
            # 06 - Evaluation and Error Analysis: {dataset_name}

            Evaluation turns a trained model into evidence. We inspect overall
            metrics, per-class behavior, confident errors, and robustness under
            simple perturbations.
            """
        ),
        common_setup(dataset_name),
        data_load_cell(),
        md(
            """
            ## 1. Load or Train a Small LeNet

            If a checkpoint from the training notebook exists, we load it. If
            not, we run a short training pass so the evaluation code remains
            executable.
            """
        ),
        code(
            """
            from mnist_fashion_research.evaluation import confusion, metrics_table, predict_batches, top_confident_errors
            from mnist_fashion_research.models import LeNet5
            from mnist_fashion_research.training import fit
            from mnist_fashion_research.utils import device, set_seed

            loaders = make_dataloaders(DATASET_NAME, batch_size=256, val_fraction=0.1, seed=SEED, augment=False, root=ROOT)
            model = LeNet5()
            checkpoint_path = ROOT / "outputs" / "models" / spec.slug / "lenet.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded {checkpoint_path}")
            else:
                print("No checkpoint found; training a short demo model.")
                fit(model, loaders, epochs=2, lr=1e-3, seed=SEED, max_train_batches=100, max_eval_batches=30)

            run_device = device()
            model = model.to(run_device)
            predictions_bundle = predict_batches(model, loaders["test"], run_device)
            labels = predictions_bundle["labels"].numpy()
            predictions = predictions_bundle["predictions"].numpy()
            probabilities = predictions_bundle["probabilities"].numpy()
            """
        ),
        md(
            """
            ## 2. Metrics and Confusion Matrix

            Overall accuracy is useful, but per-class metrics reveal whether the
            model fails systematically on specific labels.
            """
        ),
        code(
            """
            report = metrics_table(labels, predictions, class_names)
            report_df = pd.DataFrame(report).T
            display(report_df.round(4))

            cm = confusion(labels, predictions)
            plot_confusion_matrix(cm, class_names, title=f"{DATASET_NAME}: LeNet confusion matrix")
            plt.show()
            """
        ),
        md(
            """
            ## 3. Confident Errors

            Confident mistakes are valuable because they show what the model
            believes strongly but incorrectly. These examples often guide the
            next round of EDA or augmentation.
            """
        ),
        code(
            """
            error_items = top_confident_errors(X_test, labels, predictions, probabilities, top_k=25)
            if error_items:
                error_images = [item["image"] for item in error_items]
                error_labels = [item["label"] for item in error_items]
                plot_image_grid(
                    error_images,
                    error_labels,
                    class_names=class_names,
                    ncols=5,
                    title="Most confident errors: title shows true label",
                )
                plt.show()
                pd.DataFrame(
                    [
                        {
                            "test_index": item["index"],
                            "true": class_names[item["label"]],
                            "predicted": class_names[item["prediction"]],
                            "confidence": item["confidence"],
                        }
                        for item in error_items
                    ]
                ).head(10)
            else:
                print("No errors found in this evaluation run.")
            """
        ),
        code(
            """
            if error_items:
                display(
                    animate_samples(
                        [item["image"] for item in error_items],
                        [item["prediction"] for item in error_items],
                        class_names,
                        n_frames=min(20, len(error_items)),
                    )
                )
            """
        ),
        md(
            """
            ## 4. Confidence and Calibration

            A model can be accurate but poorly calibrated. Here we bin examples
            by predicted confidence and compare confidence with empirical
            accuracy.
            """
        ),
        code(
            """
            confidence = probabilities.max(axis=1)
            correct = (predictions == labels).astype(float)
            bins = np.linspace(0, 1, 11)
            bin_ids = np.digitize(confidence, bins) - 1
            calibration_rows = []
            for b in range(len(bins) - 1):
                mask = bin_ids == b
                if mask.any():
                    calibration_rows.append(
                        {
                            "bin_left": bins[b],
                            "bin_right": bins[b + 1],
                            "mean_confidence": confidence[mask].mean(),
                            "empirical_accuracy": correct[mask].mean(),
                            "count": mask.sum(),
                        }
                    )
            calibration = pd.DataFrame(calibration_rows)
            display(calibration)

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot([0, 1], [0, 1], "--", color="#777", label="perfect calibration")
            ax.plot(calibration["mean_confidence"], calibration["empirical_accuracy"], marker="o")
            ax.set_xlabel("mean predicted confidence")
            ax.set_ylabel("empirical accuracy")
            ax.set_title("Reliability diagram")
            ax.legend()
            plt.show()
            """
        ),
    ]


def interpretability_cells(dataset_name: str):
    return [
        md(
            f"""
            # 07 - Interpretability and Research Report: {dataset_name}

            Interpretability does not prove that a model thinks like a human,
            but it helps us inspect whether predictions depend on plausible
            regions of the image.
            """
        ),
        common_setup(dataset_name),
        data_load_cell(),
        md(
            """
            ## 1. Load a LeNet Model

            Use the trained checkpoint when available. Otherwise, train a short
            demo model so the saliency and occlusion examples are executable.
            """
        ),
        code(
            """
            from mnist_fashion_research.models import LeNet5
            from mnist_fashion_research.training import fit
            from mnist_fashion_research.utils import device

            loaders = make_dataloaders(DATASET_NAME, batch_size=256, val_fraction=0.1, seed=SEED, augment=False, root=ROOT)
            model = LeNet5()
            checkpoint_path = ROOT / "outputs" / "models" / spec.slug / "lenet.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                fit(model, loaders, epochs=2, lr=1e-3, seed=SEED, max_train_batches=100, max_eval_batches=30)
            run_device = device()
            model = model.to(run_device)
            model.eval()
            """
        ),
        md(
            """
            ## 2. Gradient Saliency

            Saliency asks: which pixels would most change the selected class
            score if perturbed? It is easy to compute, but can be noisy, so we
            treat it as a diagnostic view rather than final proof.
            """
        ),
        code(
            """
            raw_image = X_test[0].float() / 255.0
            normalized = (raw_image - spec.canonical_mean) / spec.canonical_std
            input_tensor = normalized.unsqueeze(0).unsqueeze(0).to(run_device)
            input_tensor.requires_grad_(True)

            logits = model(input_tensor)
            predicted_class = int(logits.argmax(dim=1).item())
            score = logits[0, predicted_class]
            model.zero_grad(set_to_none=True)
            score.backward()
            saliency = input_tensor.grad.detach().abs().cpu().squeeze().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            axes[0].imshow(raw_image.numpy(), cmap="gray")
            axes[0].set_title(f"Input: true {class_names[int(y_test[0])]}")
            axes[1].imshow(saliency, cmap="magma")
            axes[1].set_title(f"Saliency: predicted {class_names[predicted_class]}")
            for ax in axes:
                ax.axis("off")
            plt.show()
            """
        ),
        md(
            """
            ## 3. Occlusion Sensitivity

            Occlusion is slower but intuitive: hide one patch at a time and
            measure how much the predicted-class score drops.
            """
        ),
        code(
            """
            def occlusion_sensitivity(model, raw_image, predicted_class, patch=5, stride=2):
                model.eval()
                base = (raw_image - spec.canonical_mean) / spec.canonical_std
                base_input = base.unsqueeze(0).unsqueeze(0).to(run_device)
                with torch.no_grad():
                    base_score = model(base_input)[0, predicted_class].item()
                heatmap = np.zeros((28, 28), dtype=np.float32)
                counts = np.zeros((28, 28), dtype=np.float32)
                for y in range(0, 28 - patch + 1, stride):
                    for x in range(0, 28 - patch + 1, stride):
                        occluded = raw_image.clone()
                        occluded[y:y + patch, x:x + patch] = 0
                        occluded = (occluded - spec.canonical_mean) / spec.canonical_std
                        with torch.no_grad():
                            score = model(occluded.unsqueeze(0).unsqueeze(0).to(run_device))[0, predicted_class].item()
                        drop = base_score - score
                        heatmap[y:y + patch, x:x + patch] += drop
                        counts[y:y + patch, x:x + patch] += 1
                return heatmap / np.maximum(counts, 1)

            occ = occlusion_sensitivity(model, raw_image, predicted_class)
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            axes[0].imshow(raw_image.numpy(), cmap="gray")
            axes[0].set_title("Original")
            axes[1].imshow(occ, cmap="magma")
            axes[1].set_title("Occlusion sensitivity")
            for ax in axes:
                ax.axis("off")
            plt.show()
            """
        ),
        code(
            """
            from matplotlib import animation

            patch = 5
            positions = [(y, x) for y in range(0, 28 - patch + 1, 3) for x in range(0, 28 - patch + 1, 3)]
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            moving = raw_image.numpy().copy()
            image_display = axes[0].imshow(moving, cmap="gray", vmin=0, vmax=1)
            heat_display = axes[1].imshow(np.zeros((28, 28)), cmap="magma")
            for ax in axes:
                ax.axis("off")
            axes[0].set_title("Occluded input")
            axes[1].set_title("Accumulated sensitivity")
            running = np.zeros((28, 28), dtype=np.float32)

            def update(frame):
                y, x = positions[frame]
                moving = raw_image.numpy().copy()
                moving[y:y + patch, x:x + patch] = 0
                running[y:y + patch, x:x + patch] = occ[y:y + patch, x:x + patch]
                image_display.set_data(moving)
                heat_display.set_data(running)
                return image_display, heat_display

            anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=120, blit=False)
            plt.close(fig)
            HTML(anim.to_jshtml())
            """
        ),
        md(
            """
            ## 4. Research Report Checklist

            A final report should connect results back to the evidence:

            - Dataset source, split sizes, and preprocessing constants.
            - EDA findings that influenced augmentation or model choice.
            - Baseline scores and CNN scores under identical evaluation rules.
            - Training curves and selected checkpoints.
            - Confusion matrix, confident errors, calibration, and robustness.
            - Interpretability figures with careful limitations.
            """
        ),
        code(
            """
            checklist = pd.DataFrame(
                [
                    ["data audit", "EDA notebook"],
                    ["preprocessing constants", "preprocessing notebook"],
                    ["baseline comparison", "baseline notebook"],
                    ["CNN architecture justification", "LeNet/AlexNet notebook"],
                    ["training curves", "training notebook"],
                    ["final metrics and errors", "evaluation notebook"],
                    ["saliency and occlusion", "interpretability notebook"],
                ],
                columns=["report item", "source notebook"],
            )
            checklist
            """
        ),
    ]


def combined_pipeline_cells(dataset_name: str):
    return [
        md(
            f"""
            # Combined ML Pipeline: {dataset_name}

            This notebook combines the post-EDA workflow into one runnable
            pipeline: preprocessing, augmentation experiments, baseline and CNN
            model variants, training, checkpointing, evaluation, and
            visualization.

            Problem framing and exploratory data analysis are intentionally
            excluded, as requested.
            """
        ),
        common_setup(dataset_name),
        data_load_cell(),
        md(
            """
            ## 1. Preprocessing Contract

            The raw images are 28 by 28 grayscale tensors with pixel values in
            `[0, 255]`. Training transforms convert images to tensors scaled to
            `[0, 1]`, then normalize them with the canonical mean and standard
            deviation for the selected dataset.
            """
        ),
        code(
            """
            X_float = X_train.float() / 255.0
            preprocessing_summary = pd.DataFrame(
                [
                    ["train images", tuple(X_train.shape)],
                    ["test images", tuple(X_test.shape)],
                    ["computed train mean", float(X_float.mean())],
                    ["computed train std", float(X_float.std())],
                    ["canonical mean", spec.canonical_mean],
                    ["canonical std", spec.canonical_std],
                ],
                columns=["item", "value"],
            )
            preprocessing_summary
            """
        ),
        md(
            """
            ## 2. Augmentation and Split Variables

            The training grid varies image augmentation by rotation and affine
            policy. It also trains with and without stratified validation
            splitting so the effect of stratification can be compared directly.
            """
        ),
        code(
            """
            from PIL import Image
            from torchvision import transforms

            ROTATION_DEGREES = [2, 5, 10, 15]
            AFFINE_POLICIES = [
                {
                    "affine_name": "translate_4pct",
                    "affine_translate": (0.04, 0.04),
                    "affine_scale": None,
                    "affine_shear": None,
                },
                {
                    "affine_name": "translate_8pct",
                    "affine_translate": (0.08, 0.08),
                    "affine_scale": None,
                    "affine_shear": None,
                },
                {
                    "affine_name": "zoom_10pct",
                    "affine_translate": (0.04, 0.04),
                    "affine_scale": (0.90, 1.10),
                    "affine_shear": None,
                },
                {
                    "affine_name": "shear_8deg",
                    "affine_translate": (0.04, 0.04),
                    "affine_scale": None,
                    "affine_shear": (-8, 8, -8, 8),
                },
            ]
            STRATIFY_OPTIONS = [True, False]

            augmentation_table = pd.DataFrame(
                [
                    {"variable": "rotation_degrees", "values": ROTATION_DEGREES},
                    {"variable": "affine_policy", "values": [p["affine_name"] for p in AFFINE_POLICIES]},
                    {"variable": "stratified_validation_split", "values": STRATIFY_OPTIONS},
                ]
            )
            augmentation_table
            """
        ),
        code(
            """
            def augmentation_config(rotation_degrees, affine_policy):
                return {
                    "rotation_degrees": rotation_degrees,
                    "affine_translate": affine_policy["affine_translate"],
                    "affine_scale": affine_policy["affine_scale"],
                    "affine_shear": affine_policy["affine_shear"],
                }


            def preview_transform(rotation_degrees, affine_policy):
                steps = []
                if rotation_degrees:
                    steps.append(transforms.RandomRotation(rotation_degrees))
                steps.append(
                    transforms.RandomAffine(
                        degrees=0,
                        translate=affine_policy["affine_translate"],
                        scale=affine_policy["affine_scale"],
                        shear=affine_policy["affine_shear"],
                    )
                )
                steps.append(transforms.ToTensor())
                return transforms.Compose(steps)


            example_index = int(np.where(y_train_np == 0)[0][0])
            pil_example = Image.fromarray(X_train[example_index].numpy(), mode="L")
            preview_policy = AFFINE_POLICIES[1]
            preview = preview_transform(10, preview_policy)
            preview_images = [preview(pil_example).squeeze().numpy() for _ in range(16)]
            plot_image_grid(
                preview_images,
                labels=[int(y_train_np[example_index])] * len(preview_images),
                class_names=class_names,
                ncols=8,
                title=f"Augmentation preview: rotation 10 + {preview_policy['affine_name']}",
            )
            plt.show()
            """
        ),
        md(
            """
            ## 3. Model Variants and Hyperparameters

            The MLP baseline varies one hidden layer across 64, 128, and 256
            units. LeNet-style and AlexNet-style CNNs vary their convolution
            channel lists. The third CNN variant uses a deeper
            `[4, 8, 16, 32]` list because the request repeated `[4, 8, 16]`.
            """
        ),
        code(
            """
            from mnist_fashion_research.models import (
                FlexibleAlexNetMini,
                FlexibleLeNet,
                MLPBaseline,
                count_parameters,
                layer_shape_table,
            )
            from mnist_fashion_research.training import evaluate, fit, save_history
            from mnist_fashion_research.evaluation import (
                confusion,
                metrics_table,
                predict_batches,
                top_confident_errors,
            )
            from mnist_fashion_research.utils import device, ensure_dir, set_seed
            from sklearn.metrics import f1_score

            MLP_HIDDEN_LAYER_SIZES = [64, 128, 256]
            LENET_CONV_VARIANTS = [(4, 8), (4, 8, 16), (4, 8, 16, 32)]
            ALEXNET_CONV_VARIANTS = [(4, 8), (4, 8, 16), (4, 8, 16, 32)]
            EPOCHS = 4
            LR_VALUES = [5e-4, 1e-3, 2e-3]
            BATCH_SIZE_VALUES = [64, 128]
            WEIGHT_DECAY_VALUES = [0.0, 1e-4]
            HYPERPARAMETER_GRID = [
                {
                    "name": f"adam_lr{lr:g}_bs{batch_size}_wd{weight_decay:g}",
                    "epochs": EPOCHS,
                    "lr": lr,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                }
                for lr in LR_VALUES
                for batch_size in BATCH_SIZE_VALUES
                for weight_decay in WEIGHT_DECAY_VALUES
            ]
            assert len(HYPERPARAMETER_GRID) == 12


            def channel_variant_name(channels):
                return "conv_" + "_".join(str(channel) for channel in channels)


            MODEL_VARIANTS = []
            for hidden_size in MLP_HIDDEN_LAYER_SIZES:
                MODEL_VARIANTS.append(
                    {
                        "model_family": "MLP",
                        "variant": f"hidden_{hidden_size}",
                        "hidden_layer_sizes": (hidden_size,),
                        "conv_channels": None,
                    }
                )
            for channels in LENET_CONV_VARIANTS:
                MODEL_VARIANTS.append(
                    {
                        "model_family": "LeNet",
                        "variant": channel_variant_name(channels),
                        "hidden_layer_sizes": None,
                        "conv_channels": channels,
                    }
                )
            for channels in ALEXNET_CONV_VARIANTS:
                MODEL_VARIANTS.append(
                    {
                        "model_family": "AlexNet",
                        "variant": channel_variant_name(channels),
                        "hidden_layer_sizes": None,
                        "conv_channels": channels,
                    }
                )


            def make_model(experiment):
                family = experiment["model_family"]
                if family == "MLP":
                    return MLPBaseline(hidden_layer_sizes=tuple(experiment["hidden_layer_sizes"]), dropout=0.2)
                if family == "LeNet":
                    return FlexibleLeNet(conv_channels=tuple(experiment["conv_channels"]))
                if family == "AlexNet":
                    return FlexibleAlexNetMini(conv_channels=tuple(experiment["conv_channels"]), dropout=0.3)
                raise ValueError(f"Unknown model family: {family}")


            model_catalog = []
            for variant in MODEL_VARIANTS:
                model = make_model({**variant})
                model_catalog.append(
                    {
                        "model_family": variant["model_family"],
                        "variant": variant["variant"],
                        "hidden_layer_sizes": variant["hidden_layer_sizes"],
                        "conv_channels": variant["conv_channels"],
                        "trainable_parameters": count_parameters(model),
                    }
                )
            model_catalog = pd.DataFrame(model_catalog)
            model_catalog.sort_values("trainable_parameters")
            """
        ),
        code(
            """
            fig, ax = plt.subplots(figsize=(10, 4))
            ordered_catalog = model_catalog.sort_values("trainable_parameters")
            labels_for_plot = ordered_catalog["model_family"] + " " + ordered_catalog["variant"]
            ax.bar(labels_for_plot, ordered_catalog["trainable_parameters"], color="#4C78A8")
            ax.set_yscale("log")
            ax.set_ylabel("trainable parameters (log scale)")
            ax.set_title(f"{DATASET_NAME}: model capacity comparison")
            ax.tick_params(axis="x", rotation=45)
            plt.show()
            """
        ),
        code(
            """
            pd.DataFrame(layer_shape_table(FlexibleLeNet(conv_channels=(4, 8, 16))))
            """
        ),
        md(
            """
            ## 4. Experiment Grid

            The full grid contains every requested factor combination. The
            default `balanced` run covers each factor without training the full
            Cartesian product. Change `RUN_MODE` to `"full_grid"` for the
            exhaustive sweep.
            """
        ),
        code(
            """
            BASE_ROTATION = 10
            BASE_AFFINE_POLICY = AFFINE_POLICIES[1]
            BASE_STRATIFY = True
            BASE_HPARAMS = HYPERPARAMETER_GRID[0]
            BASE_MODEL_VARIANT = next(v for v in MODEL_VARIANTS if v["model_family"] == "LeNet" and v["variant"] == "conv_4_8")


            def experiment_record(model_variant, rotation, affine_policy, stratify, hparams, purpose):
                experiment_id = "_".join(
                    [
                        model_variant["model_family"].lower(),
                        model_variant["variant"],
                        f"rot{rotation}",
                        affine_policy["affine_name"],
                        "stratified" if stratify else "randomsplit",
                        hparams["name"],
                    ]
                )
                return {
                    "experiment_id": experiment_id,
                    "purpose": purpose,
                    "model_family": model_variant["model_family"],
                    "variant": model_variant["variant"],
                    "hidden_layer_sizes": model_variant["hidden_layer_sizes"],
                    "conv_channels": model_variant["conv_channels"],
                    "rotation_degrees": rotation,
                    "affine_name": affine_policy["affine_name"],
                    "affine_policy": affine_policy,
                    "stratify": stratify,
                    **hparams,
                }


            full_experiments = [
                experiment_record(model_variant, rotation, affine_policy, stratify, hparams, "full_grid")
                for model_variant in MODEL_VARIANTS
                for rotation in ROTATION_DEGREES
                for affine_policy in AFFINE_POLICIES
                for stratify in STRATIFY_OPTIONS
                for hparams in HYPERPARAMETER_GRID
            ]

            balanced_experiments = []
            balanced_experiments.extend(
                experiment_record(BASE_MODEL_VARIANT, rotation, BASE_AFFINE_POLICY, BASE_STRATIFY, BASE_HPARAMS, "rotation_sweep")
                for rotation in ROTATION_DEGREES
            )
            balanced_experiments.extend(
                experiment_record(BASE_MODEL_VARIANT, BASE_ROTATION, affine_policy, BASE_STRATIFY, BASE_HPARAMS, "affine_sweep")
                for affine_policy in AFFINE_POLICIES
            )
            balanced_experiments.extend(
                experiment_record(BASE_MODEL_VARIANT, BASE_ROTATION, BASE_AFFINE_POLICY, stratify, BASE_HPARAMS, "stratification_sweep")
                for stratify in STRATIFY_OPTIONS
            )
            balanced_experiments.extend(
                experiment_record(model_variant, BASE_ROTATION, BASE_AFFINE_POLICY, BASE_STRATIFY, hparams, "model_hparam_sweep")
                for model_variant in MODEL_VARIANTS
                for hparams in HYPERPARAMETER_GRID
            )


            def dedupe_experiments(experiments):
                deduped = {}
                for experiment in experiments:
                    deduped.setdefault(experiment["experiment_id"], experiment)
                return list(deduped.values())


            RUN_MODE = "balanced"  # Choose "balanced" or "full_grid".
            EXPERIMENTS_TO_RUN = full_experiments if RUN_MODE == "full_grid" else dedupe_experiments(balanced_experiments)

            experiment_grid = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in experiment.items()
                        if key not in {"affine_policy"}
                    }
                    for experiment in EXPERIMENTS_TO_RUN
                ]
            )
            print(f"Full grid size: {len(full_experiments)}")
            print(f"Selected {RUN_MODE} experiments: {len(EXPERIMENTS_TO_RUN)}")
            experiment_grid.head(20)
            """
        ),
        md(
            """
            ## 5. Train, Evaluate, and Save Checkpoints

            Each selected experiment saves a best-validation checkpoint and a
            JSON training history. For quick iteration the batch limits are set
            below; set them to `None` for a full training run.
            """
        ),
        code(
            """
            MAX_TRAIN_BATCHES = 40
            MAX_EVAL_BATCHES = 20
            VAL_FRACTION = 0.1

            checkpoint_dir = ROOT / "outputs" / "models" / spec.slug / "combined_pipeline"
            report_dir = ROOT / "outputs" / "reports" / spec.slug
            ensure_dir(checkpoint_dir)
            ensure_dir(report_dir)

            criterion = torch.nn.CrossEntropyLoss()
            run_device = device()
            results = []
            histories = {}
            experiment_lookup = {experiment["experiment_id"]: experiment for experiment in EXPERIMENTS_TO_RUN}

            for run_index, experiment in enumerate(EXPERIMENTS_TO_RUN, start=1):
                set_seed(SEED + run_index)
                print(f"[{run_index:03d}/{len(EXPERIMENTS_TO_RUN):03d}] {experiment['experiment_id']}")
                loaders = make_dataloaders(
                    DATASET_NAME,
                    batch_size=experiment["batch_size"],
                    val_fraction=VAL_FRACTION,
                    seed=SEED,
                    augment=True,
                    stratify=experiment["stratify"],
                    augmentation_config=augmentation_config(experiment["rotation_degrees"], experiment["affine_policy"]),
                    root=ROOT,
                )
                model = make_model(experiment)
                trainable_parameters = count_parameters(model)
                checkpoint_path = checkpoint_dir / f"{experiment['experiment_id']}.pt"
                history_path = report_dir / f"{experiment['experiment_id']}_history.json"

                history = fit(
                    model,
                    loaders,
                    epochs=experiment["epochs"],
                    lr=experiment["lr"],
                    weight_decay=experiment["weight_decay"],
                    seed=SEED + run_index,
                    checkpoint_path=checkpoint_path,
                    max_train_batches=MAX_TRAIN_BATCHES,
                    max_eval_batches=MAX_EVAL_BATCHES,
                )
                save_history(history, history_path)
                histories[experiment["experiment_id"]] = history

                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                checkpoint["experiment"] = {
                    key: value
                    for key, value in experiment.items()
                    if key not in {"affine_policy"}
                }
                checkpoint["trainable_parameters"] = trainable_parameters
                torch.save(checkpoint, checkpoint_path)

                model.load_state_dict(checkpoint["model_state_dict"])
                model = model.to(run_device)
                predictions_bundle = predict_batches(model, loaders["test"], run_device, max_batches=MAX_EVAL_BATCHES)
                labels = predictions_bundle["labels"].numpy()
                predictions = predictions_bundle["predictions"].numpy()

                results.append(
                    {
                        "experiment_id": experiment["experiment_id"],
                        "purpose": experiment["purpose"],
                        "model_family": experiment["model_family"],
                        "variant": experiment["variant"],
                        "rotation_degrees": experiment["rotation_degrees"],
                        "affine_name": experiment["affine_name"],
                        "stratify": experiment["stratify"],
                        "epochs": experiment["epochs"],
                        "lr": experiment["lr"],
                        "batch_size": experiment["batch_size"],
                        "weight_decay": experiment["weight_decay"],
                        "trainable_parameters": trainable_parameters,
                        "best_val_accuracy": max(row["val_accuracy"] for row in history),
                        "final_val_accuracy": history[-1]["val_accuracy"],
                        "test_accuracy": float((predictions == labels).mean()),
                        "test_macro_f1": f1_score(labels, predictions, average="macro"),
                        "checkpoint_path": str(checkpoint_path),
                        "history_path": str(history_path),
                    }
                )

            results_df = pd.DataFrame(results).sort_values("best_val_accuracy", ascending=False)
            results_path = report_dir / f"{spec.slug}_combined_pipeline_results.csv"
            results_df.to_csv(results_path, index=False)
            print(f"Saved results to {results_path}")
            results_df.head(20)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            top_results = results_df.sort_values("test_accuracy", ascending=False).head(15)
            axes[0].barh(top_results["experiment_id"][::-1], top_results["test_accuracy"][::-1], color="#4C78A8")
            axes[0].set_xlim(0, 1)
            axes[0].set_xlabel("test accuracy")
            axes[0].set_title("Top checkpoint accuracy")

            for family, group in results_df.groupby("model_family"):
                axes[1].scatter(
                    group["trainable_parameters"],
                    group["test_accuracy"],
                    s=70,
                    alpha=0.8,
                    label=family,
                )
            axes[1].set_xscale("log")
            axes[1].set_ylim(0, 1)
            axes[1].set_xlabel("trainable parameters (log scale)")
            axes[1].set_ylabel("test accuracy")
            axes[1].set_title("Accuracy versus model size")
            axes[1].legend()

            fig.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            comparison_views = []
            for purpose in ["rotation_sweep", "affine_sweep", "stratification_sweep", "model_hparam_sweep"]:
                subset = results_df[results_df["purpose"] == purpose]
                if not subset.empty:
                    comparison_views.append(
                        subset.sort_values("test_accuracy", ascending=False).head(10)
                    )
            display(pd.concat(comparison_views, ignore_index=True))

            rotation_view = results_df[results_df["purpose"] == "rotation_sweep"]
            if not rotation_view.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(rotation_view["rotation_degrees"], rotation_view["test_accuracy"], marker="o")
                ax.set_xlabel("rotation degrees")
                ax.set_ylabel("test accuracy")
                ax.set_ylim(0, 1)
                ax.set_title("Rotation sweep")
                plt.show()

            affine_view = results_df[results_df["purpose"] == "affine_sweep"]
            if not affine_view.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(affine_view["affine_name"], affine_view["test_accuracy"], color="#F58518")
                ax.set_ylim(0, 1)
                ax.set_ylabel("test accuracy")
                ax.set_title("Affine policy sweep")
                ax.tick_params(axis="x", rotation=30)
                plt.show()
            """
        ),
        md(
            """
            ## 6. Best Model Evaluation

            The best validation checkpoint is loaded for a more detailed test
            report, including a confusion matrix and confident error examples.
            """
        ),
        code(
            """
            best_row = results_df.iloc[0]
            best_experiment = experiment_lookup[best_row["experiment_id"]]
            best_model = make_model(best_experiment)
            checkpoint = torch.load(best_row["checkpoint_path"], map_location="cpu", weights_only=False)
            best_model.load_state_dict(checkpoint["model_state_dict"])
            best_model = best_model.to(run_device)

            final_loaders = make_dataloaders(
                DATASET_NAME,
                batch_size=256,
                val_fraction=VAL_FRACTION,
                seed=SEED,
                augment=False,
                stratify=best_experiment["stratify"],
                root=ROOT,
            )
            predictions_bundle = predict_batches(best_model, final_loaders["test"], run_device)
            labels = predictions_bundle["labels"].numpy()
            predictions = predictions_bundle["predictions"].numpy()
            probabilities = predictions_bundle["probabilities"].numpy()

            report_df = pd.DataFrame(metrics_table(labels, predictions, class_names)).T
            display(best_row.to_frame(name="best_experiment"))
            display(report_df.round(4))

            cm = confusion(labels, predictions)
            plot_confusion_matrix(
                cm,
                class_names,
                title=f"{DATASET_NAME}: best combined-pipeline checkpoint",
            )
            plt.show()
            """
        ),
        code(
            """
            error_items = top_confident_errors(X_test[: len(labels)], labels, predictions, probabilities, top_k=25)
            if error_items:
                plot_image_grid(
                    [item["image"] for item in error_items],
                    [item["label"] for item in error_items],
                    class_names=class_names,
                    ncols=5,
                    title="Most confident errors: title shows true label",
                )
                plt.show()
                display(
                    pd.DataFrame(
                        [
                            {
                                "test_index": item["index"],
                                "true": class_names[item["label"]],
                                "predicted": class_names[item["prediction"]],
                                "confidence": item["confidence"],
                            }
                            for item in error_items
                        ]
                    ).head(10)
                )
            else:
                print("No errors found for the best checkpoint.")
            """
        ),
        md(
            """
            ## 7. Training Curves and Interpretability Check

            These cells visualize the best run's learning curve and produce a
            simple gradient saliency map for the selected checkpoint.
            """
        ),
        code(
            """
            best_history = histories.get(best_row["experiment_id"])
            if best_history is None:
                best_history = checkpoint["history"]
            best_history_df = pd.DataFrame(best_history)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(best_history_df["epoch"], best_history_df["train_loss"], marker="o", label="train")
            axes[0].plot(best_history_df["epoch"], best_history_df["val_loss"], marker="o", label="validation")
            axes[0].set_title("Best run loss")
            axes[0].legend()
            axes[1].plot(best_history_df["epoch"], best_history_df["train_accuracy"], marker="o", label="train")
            axes[1].plot(best_history_df["epoch"], best_history_df["val_accuracy"], marker="o", label="validation")
            axes[1].set_ylim(0, 1)
            axes[1].set_title("Best run accuracy")
            axes[1].legend()
            plt.show()

            display(animate_training_curve(best_history))
            """
        ),
        code(
            """
            best_model.eval()
            raw_image = X_test[0].float() / 255.0
            normalized = (raw_image - spec.canonical_mean) / spec.canonical_std
            input_tensor = normalized.unsqueeze(0).unsqueeze(0).to(run_device)
            input_tensor.requires_grad_(True)

            logits = best_model(input_tensor)
            predicted_class = int(logits.argmax(dim=1).item())
            score = logits[0, predicted_class]
            best_model.zero_grad(set_to_none=True)
            score.backward()
            saliency = input_tensor.grad.detach().abs().cpu().squeeze().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            axes[0].imshow(raw_image.numpy(), cmap="gray")
            axes[0].set_title(f"Input: true {class_names[int(y_test[0])]}")
            axes[1].imshow(saliency, cmap="magma")
            axes[1].set_title(f"Saliency: predicted {class_names[predicted_class]}")
            for ax in axes:
                ax.axis("off")
            plt.show()
            """
        ),
    ]


STEP_BUILDERS = [
    ("00_problem_framing", problem_framing_cells),
    ("01_EDA", eda_cells),
    ("02_preprocessing_augmentation", preprocessing_cells),
    ("03_baselines", baseline_cells),
    ("04_lenet_alexnet", lenet_alexnet_cells),
    ("05_training_pipeline", training_cells),
    ("06_evaluation_error_analysis", evaluation_cells),
    ("07_interpretability_report", interpretability_cells),
]


def write_notebook(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(notebook, path)


def main() -> None:
    for step, _ in STEP_BUILDERS:
        legacy_path = ROOT / "notebooks" / step
        if legacy_path.exists():
            shutil.rmtree(legacy_path)
            print(f"Removed {legacy_path.relative_to(ROOT)}")

    for dataset_name in DATASET_SPECS:
        notebook_name = f"combined_pipeline_{dataset_name}.ipynb"
        path = ROOT / "notebooks" / dataset_name / notebook_name
        write_notebook(path, combined_pipeline_cells(dataset_name))
        print(f"Wrote {path.relative_to(ROOT)}")

    readme = ROOT / "notebooks" / "README.md"
    readme.write_text(
        dedent(
            """
            # Notebooks

            The notebooks are organized by dataset. Each dataset folder contains
            one combined pipeline notebook.

            Layout:

            - `MNIST/combined_pipeline_MNIST.ipynb`
            - `FashionMNIST/combined_pipeline_FashionMNIST.ipynb`

            The combined notebooks start after problem framing and EDA. They
            include preprocessing, augmentation sweeps, stratified and
            unstratified training, MLP baseline sweeps, LeNet/AlexNet-style CNN
            sweeps, checkpoint saving, and evaluation visualizations.
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
