"""Cấu hình @dataclass CFG & Bộ hàm EDA chữ ngắn gọn tách rời (dùng chung Folder & CSV)."""
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TORCH_HOME", "/opt/torch_models")


@dataclass
class CFG:
    SEED: int = 42
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LR: float = 3e-4
    VAL_RATIO: float = 0.2
    PATIENCE: int = 5
    TRAIN_DIR: str = "./dataset/train"
    TEST_DIR: str = "./dataset/test"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scan_folder_samples(root_dir: str) -> tuple[list[Path], list[str]]:
    """Quét thư mục dạng root/class_name/* trả về (danh_sách_ảnh, danh_sách_nhãn)."""
    paths, labels = [], []
    for cls_dir in sorted(p for p in Path(root_dir).iterdir() if p.is_dir()):
        for img_p in sorted(p for p in cls_dir.iterdir() if p.is_file()):
            paths.append(img_p)
            labels.append(cls_dir.name)
    return paths, labels


def check_image_stats(img_paths: list[str | Path], sample_k: int = 200) -> dict:
    """Kiểm tra nhanh tệp hỏng, hệ màu và kích thước (min, median, max) trên mẫu ngẫu nhiên."""
    sampled = random.sample(list(img_paths), k=min(sample_k, len(img_paths)))
    ws, hs, modes, broken = [], [], Counter(), 0
    for p in sampled:
        try:
            with Image.open(p) as im:
                ws.append(im.width)
                hs.append(im.height)
                modes[im.mode] += 1
        except Exception:
            broken += 1
    return {
        "total_images": len(img_paths),
        "broken_in_sample": broken,
        "modes": dict(modes),
        "width_min_med_max": (int(np.min(ws)), int(np.median(ws)), int(np.max(ws))) if ws else (),
        "height_min_med_max": (int(np.min(hs)), int(np.median(hs)), int(np.max(hs))) if hs else (),
    }


def check_label_balance(labels: list) -> dict:
    """Thống kê số lớp và tỷ lệ lệch lớp (min, max)."""
    cnt = Counter(labels)
    return {
        "num_classes": len(cnt),
        "class_min_max": (min(cnt.values()), max(cnt.values())) if cnt else (),
    }


def quick_cv_eda(img_paths: list[str | Path], labels: list | None = None, sample_k: int = 200) -> dict:
    """Gộp thống kê kích thước ảnh và phân phối nhãn chỉ trong 1 lời gọi."""
    info = check_image_stats(img_paths, sample_k=sample_k)
    if labels is not None:
        info.update(check_label_balance(labels))
    print("[CV EDA]", info)
    return info
