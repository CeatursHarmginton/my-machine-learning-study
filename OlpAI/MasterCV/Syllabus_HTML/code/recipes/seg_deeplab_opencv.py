import cv2
import numpy as np
import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

# 1. Mô hình DeepLabV3 tích hợp ASPP đa tỷ lệ dựng sẵn trong torchvision (hỗ trợ mọi số lớp num_classes):
def build_deeplabv3(num_classes: int = 3) -> nn.Module:
    net = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    net.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return net

# 2. Hậu xử lý xóa các đốm nhiễu rời rạc diện tích < min_area bằng OpenCV (giảm mạnh lỗi Hausdorff!):
def remove_small_blobs_cv2(prob_map: np.ndarray, thresh: float = 0.5, min_area: int = 120) -> np.ndarray:
    mask = (prob_map >= thresh).astype(np.uint8)
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.isin(labels, [i for i in range(1, n_lbl) if stats[i, cv2.CC_STAT_AREA] >= min_area])
    return keep.astype(np.uint8)
