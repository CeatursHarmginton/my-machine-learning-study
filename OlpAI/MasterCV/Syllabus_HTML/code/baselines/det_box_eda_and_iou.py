"""Bộ hàm EDA Hộp Bao, Lọc Hộp Suy Biến & Tính IoU / NMS bằng NumPy."""
import numpy as np
import pandas as pd


def find_invalid_boxes(df: pd.DataFrame) -> pd.Series:
    return (df["x_max"] <= df["x_min"]) | (df["y_max"] <= df["y_min"])


def filter_valid_boxes(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[~find_invalid_boxes(df)].reset_index(drop=True)


def quick_bbox_eda(df: pd.DataFrame) -> dict:
    per_img = df.groupby("image_id").size()
    info = {
        "total_boxes": len(df),
        "num_images": int(per_img.shape[0]),
        "invalid_boxes": int(find_invalid_boxes(df).sum()),
        "boxes_per_img_min_med_max": (int(per_img.min()), int(per_img.median()), int(per_img.max())),
    }
    print("[BBox EDA]", info)
    return info


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    a1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    a2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / np.maximum(a1[:, None] + a2[None, :] - inter, 1e-8)


def greedy_nms_np(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> list[int]:
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_np(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return keep
