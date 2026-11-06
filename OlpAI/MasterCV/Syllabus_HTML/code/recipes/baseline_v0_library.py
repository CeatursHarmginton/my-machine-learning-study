import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from torchvision import models

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_cv_folds(df: pd.DataFrame, y_col: str, group_col: str | None = None,
                  domain_col: str | None = None, n_splits: int = 5, seed: int = 42):
    """Chia Validation tổng quát bằng scikit-learn: tự động chọn chuẩn theo cấu trúc rò rỉ dữ liệu."""
    y = df[y_col].to_numpy()
    if domain_col:
        return list(LeaveOneGroupOut().split(df, y, df[domain_col].to_numpy()))
    if group_col:
        return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(df, y, df[group_col].to_numpy()))
    return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, y))

def build_torchvision_clf(arch: str = "resnet34", num_classes: int = 5) -> nn.Module:
    """Khởi tạo tổng quát mọi kiến trúc trong torchvision.models (ResNet / ConvNeXt / Swin / ViT)."""
    net = getattr(models, arch)(weights="DEFAULT")
    if hasattr(net, "fc"):
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif hasattr(net, "classifier") and isinstance(net.classifier, nn.Sequential):
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
    elif hasattr(net, "heads"):
        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
    return net
