"""Tách nhỏ các hàm phân tách nhãn BIO, gom thực thể (Span) & tính điểm Strict Span-F1."""


def parse_bio_tag(tag: str) -> tuple[str, str]:
    if "-" in tag:
        prefix, ent = tag.split("-", 1)
        return prefix, ent
    return "O", ""


def decode_bio_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    spans, cur_type, start = set(), None, -1
    for i, tag in enumerate(tags):
        prefix, ent = parse_bio_tag(tag)
        if prefix == "B":
            if cur_type:
                spans.add((cur_type, start, i - 1))
            cur_type, start = ent, i
        elif prefix == "I":
            if cur_type != ent:
                if cur_type:
                    spans.add((cur_type, start, i - 1))
                cur_type, start = ent, i
        elif cur_type:
            spans.add((cur_type, start, i - 1))
            cur_type, start = None, -1
    if cur_type:
        spans.add((cur_type, start, len(tags) - 1))
    return spans


def count_span_overlap(pred_spans: set, gold_spans: set) -> tuple[int, int, int]:
    return len(pred_spans & gold_spans), len(pred_spans - gold_spans), len(gold_spans - pred_spans)


def compute_strict_span_f1(pred_seqs: list[list[str]], gold_seqs: list[list[str]]) -> float:
    tp = fp = fn = 0
    for p, g in zip(pred_seqs, gold_seqs):
        t_i, f_p, f_n = count_span_overlap(decode_bio_spans(p), decode_bio_spans(g))
        tp, fp, fn = tp + t_i, fp + f_p, fn + f_n
    return round(100.0 * (2 * tp) / max(2 * tp + fp + fn, 1), 2)
