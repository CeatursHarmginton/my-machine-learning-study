import math
import torch
import torch.nn as nn
from torchvision.ops import box_iou

def init_rare_prior_bias(cls_conv: nn.Conv2d, prior_prob: float = 0.01) -> None:
    """Ngăn hàng vạn vùng nền đè bẹp đạo hàm ở epoch 1: khởi tạo b = -log((1 - pi) / pi) ~ -4.595."""
    nn.init.normal_(cls_conv.weight, std=0.01)
    nn.init.constant_(cls_conv.bias, -math.log((1.0 - prior_prob) / prior_prob))

def weighted_boxes_fusion_torch(boxes_list: list[torch.Tensor], scores_list: list[torch.Tensor],
                                labels_list: list[torch.Tensor], iou_thr: float = 0.55):
    """WBF đa lớp tổng quát 22 dòng bằng PyTorch + box_iou (thay thế hoàn toàn gói ngoài ensemble_boxes!)."""
    n_models = len(boxes_list)
    boxes, scores, labels = torch.cat(boxes_list), torch.cat(scores_list), torch.cat(labels_list)
    out_b, out_s, out_l = [], [], []
    for cls_id in labels.unique():
        m = labels == cls_id
        b_c, s_c = boxes[m], scores[m]
        order = s_c.argsort(descending=True)
        b_c, s_c = b_c[order], s_c[order]
        clusters_b, clusters_s, fused = [], [], []
        for b, s in zip(b_c, s_c):
            ious = box_iou(b[None], torch.stack(fused))[0] if fused else torch.empty(0, device=b.device)
            if len(ious) > 0 and ious.max() >= iou_thr:
                k = int(ious.argmax())
                clusters_b[k].append(b)
                clusters_s[k].append(s)
                w = torch.stack(clusters_s[k])[:, None]
                fused[k] = (w * torch.stack(clusters_b[k])).sum(0) / w.sum()
            else:
                clusters_b.append([b])
                clusters_s.append([s])
                fused.append(b)
        out_b.extend(fused)
        out_s.extend([torch.stack(cs).sum() / n_models for cs in clusters_s])
        out_l.extend([cls_id] * len(fused))
    return torch.stack(out_b), torch.stack(out_s), torch.stack(out_l)
