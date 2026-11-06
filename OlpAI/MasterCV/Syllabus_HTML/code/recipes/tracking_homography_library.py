import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.ops import box_iou

# 1. Phát hiện vật thể bằng Faster R-CNN v2 (torchvision) + Ghép quỹ đạo Hungarian bằng SciPy (không cần ultralytics!):
detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).cuda().eval()

def match_detections_scipy(prev_boxes: torch.Tensor, cur_boxes: torch.Tensor, iou_thr: float = 0.3):
    iou = box_iou(prev_boxes, cur_boxes).cpu().numpy()
    rows, cols = linear_sum_assignment(1.0 - iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= iou_thr]

# 2. Ước lượng ma trận chiếu Homography (3, 3) kháng ngoại lai bằng OpenCV USAC_MAGSAC / RANSAC:
def align_plane_opencv(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_mat, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.USAC_MAGSAC, ransacReprojThreshold=4.0)
    return h_mat, mask.ravel().astype(bool)
