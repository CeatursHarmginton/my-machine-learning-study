import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

def _hungarian_iou(tracks: torch.Tensor, dets: torch.Tensor, iou_th: float):
    if len(tracks) == 0 or len(dets) == 0:
        return [], list(range(len(tracks))), list(range(len(dets)))
    iou = box_iou(tracks, dets).cpu().numpy()
    rs, cs = linear_sum_assignment(1.0 - iou)
    m = [(int(r), int(c)) for r, c in zip(rs, cs) if iou[r, c] >= iou_th]
    return m, sorted(set(range(len(tracks))) - {r for r, _ in m}), sorted(set(range(len(dets))) - {c for _, c in m})

def bytetrack_two_stage(pred_tracks: torch.Tensor, dets: torch.Tensor, scores: torch.Tensor, hi: float = 0.5, lo: float = 0.1):
    """Ghép cặp 2 pha của ByteTrack: pha 1 khớp hộp điểm cao, pha 2 cứu vật thể bị che khuất bằng hộp điểm thấp!"""
    hi_ix, lo_ix = torch.where(scores >= hi)[0], torch.where((scores >= lo) & (scores < hi))[0]
    m1, rem_t, new_d = _hungarian_iou(pred_tracks, dets[hi_ix], iou_th=0.3)
    m2, lost_t, _ = _hungarian_iou(pred_tracks[rem_t], dets[lo_ix], iou_th=0.4)
    return [(t, int(hi_ix[d])) for t, d in m1] + [(rem_t[t], int(lo_ix[d])) for t, d in m2], [rem_t[t] for t in lost_t], [int(hi_ix[d]) for d in new_d]

def solve_dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Giải ma trận chiếu H (3x3) bằng chuẩn hóa Hartley + phân rã SVD của ma trận A (2N x 9)."""
    def _norm(p):
        c = p.mean(0)
        s = np.sqrt(2.0) / (np.linalg.norm(p - c, axis=1).mean() + 1e-8)
        t = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
        return (p - c) * s, t
    sn, ts = _norm(src)
    dn, td = _norm(dst)
    x, y, u, v = sn[:, 0], sn[:, 1], dn[:, 0], dn[:, 1]
    z, o = np.zeros(len(sn)), np.ones(len(sn))
    a_mat = np.vstack([np.column_stack([-x, -y, -o, z, z, z, u * x, u * y, u]),
                       np.column_stack([z, z, z, -x, -y, -o, v * x, v * y, v])])
    h_mat = np.linalg.inv(td) @ np.linalg.svd(a_mat)[2][-1].reshape(3, 3) @ ts
    return h_mat / (h_mat[2, 2] + 1e-12)
