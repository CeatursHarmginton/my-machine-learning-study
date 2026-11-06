def rrf_fuse(rank_lists: list[list[int]], weights: list[float] | None = None, k: int = 60, top_n: int = 50) -> list[int]:
    """Reciprocal Rank Fusion (9 dòng): hợp nhất BM25 và Dense KNN không phụ thuộc thang điểm tuyệt đối."""
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[int, float] = {}
    for r_list, w in zip(rank_lists, weights):
        for rank, doc_id in enumerate(r_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]

def clean_hard_negatives(pos_score: float, cand_ids: list[int], cand_scores: list[float],
                         gold_ids: set[int], max_ratio: float = 0.92) -> list[int]:
    """Loại bỏ False Negatives (ứng viên chưa gán nhãn nhưng có điểm Cross-Encoder sát mẫu dương thật)."""
    return [cid for cid, sc in zip(cand_ids, cand_scores) if cid not in gold_ids and sc < max_ratio * pos_score]
