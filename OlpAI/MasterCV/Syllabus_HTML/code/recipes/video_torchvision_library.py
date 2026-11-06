import torch
import torch.nn as nn
from torchvision.models.video import Swin3D_T_Weights, swin3d_t

def build_video_classifier(num_classes: int) -> nn.Module:
    """Tải mạng Swin3D-Tiny tiền huấn luyện Kinetics-400 từ torchvision.models.video, đầu vào (B, C, T, H, W)."""
    net = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)
    net.head = nn.Linear(net.head.in_features, num_classes)
    return net

# Kiểm tra luồng suy luận với clip 16 khung hình 224x224:
video_net = build_video_classifier(num_classes=4).cuda().eval()
with torch.no_grad():
    logits = video_net(torch.randn(2, 3, 16, 224, 224, device="cuda"))
