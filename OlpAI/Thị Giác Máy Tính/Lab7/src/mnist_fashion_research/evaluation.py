from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


@torch.no_grad()
def predict_batches(model, loader, device):
    model.eval()
    logits_parts = []
    label_parts = []
    for images, labels in loader:
        images = images.to(device)
        logits_parts.append(model(images).cpu())
        label_parts.append(labels.cpu())
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts)
    probabilities = logits.softmax(dim=1)
    predictions = probabilities.argmax(dim=1)
    return {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "labels": labels,
    }


def metrics_table(labels, predictions, class_names):
    return classification_report(
        np.asarray(labels),
        np.asarray(predictions),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )


def confusion(labels, predictions):
    return confusion_matrix(np.asarray(labels), np.asarray(predictions))


def top_confident_errors(images, labels, predictions, probabilities, top_k: int = 25):
    labels = torch.as_tensor(labels)
    predictions = torch.as_tensor(predictions)
    probabilities = torch.as_tensor(probabilities)
    wrong = predictions != labels
    if wrong.sum() == 0:
        return []
    confidence = probabilities.max(dim=1).values
    indices = torch.where(wrong)[0]
    ranked = indices[torch.argsort(confidence[indices], descending=True)]
    result = []
    for index in ranked[:top_k]:
        result.append(
            {
                "index": int(index),
                "image": images[int(index)],
                "label": int(labels[int(index)]),
                "prediction": int(predictions[int(index)]),
                "confidence": float(confidence[int(index)]),
            }
        )
    return result

