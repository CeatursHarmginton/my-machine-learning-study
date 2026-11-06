import torch
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import batched_nms, generalized_box_iou_loss

# 1. Khởi tạo RetinaNet-FPN-v2 của torchvision cho số lớp bất kỳ (tự động gắn sẵn Prior Bias chống sụp đổ!):
def build_retinanet_detector(num_classes: int = 15):
    det = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    n_anchors = det.head.classification_head.num_anchors
    det.head.classification_head = RetinaNetClassificationHead(256, n_anchors, num_classes)
    return det

# 2. Lọc hộp bao đa lớp chuẩn xác bằng batched_nms & GIoU Loss dựng sẵn trong torchvision.ops:
def postprocess_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thr: float = 0.5):
    keep = batched_nms(boxes, scores, labels, iou_threshold=iou_thr)
    return boxes[keep], scores[keep], labels[keep]
