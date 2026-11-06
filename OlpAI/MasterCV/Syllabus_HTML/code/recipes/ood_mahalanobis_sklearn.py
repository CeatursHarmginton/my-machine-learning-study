import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve

class SklearnMahalanobisOOD:
    """Phát hiện OOD tổng quát cho mọi tập nhãn bằng LedoitWolf (chống suy biến khi số mẫu < số chiều D!)."""
    def fit(self, train_feats: np.ndarray, train_labels: np.ndarray):
        classes = np.unique(train_labels)
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.means = np.vstack([train_feats[train_labels == c].mean(axis=0) for c in classes])
        centered = train_feats - self.means[[cls_to_idx[y] for y in train_labels]]
        self.precision = LedoitWolf().fit(centered).precision_
        return self

    def ood_score(self, test_feats: np.ndarray) -> np.ndarray:
        diff = test_feats[:, None, :] - self.means[None, :, :]
        return np.einsum("ncd,de,nce->nc", diff, self.precision, diff).min(axis=1)

def eval_ood_metrics_sklearn(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict[str, float]:
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    fpr, tpr, _ = roc_curve(y_true, scores)
    return {"auroc": round(float(roc_auc_score(y_true, scores)), 4), "fpr95": round(float(fpr[np.searchsorted(tpr, 0.95)]), 4)}
