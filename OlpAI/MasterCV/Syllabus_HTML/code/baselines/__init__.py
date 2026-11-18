"""MasterCV baselines subpackage: Classification, Video CRNN, Faster R-CNN Detection & U-Net Segmentation."""
from .cv_eda_and_config import CFG, check_image_stats, check_label_balance, quick_cv_eda, seed_everything

__all__ = ["CFG", "seed_everything", "quick_cv_eda", "check_image_stats", "check_label_balance"]
