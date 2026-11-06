import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# Tải trọng số MCG-NJU/videomae-base đã tiền huấn luyện bằng Tube Masking 90% trong transformers:
ckpt = "MCG-NJU/videomae-base"
vmae_proc = VideoMAEImageProcessor.from_pretrained(ckpt)
vmae_net = VideoMAEForVideoClassification.from_pretrained(ckpt, num_labels=10, ignore_mismatched_sizes=True).cuda()

frames = [torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy() for _ in range(16)]
inputs = vmae_proc(frames, return_tensors="pt").to("cuda")
pred_cls = vmae_net(**inputs).logits.argmax(dim=-1)
