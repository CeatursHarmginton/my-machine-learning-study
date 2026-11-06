from transformers import AutoModelForTokenClassification, AutoTokenizer

def align_spans_to_bio(tok, text: str, spans: list[tuple[int, int, str]], label2id: dict[str, int], max_len: int = 512):
    """Căn chỉnh mọi danh sách khoảng ký tự (start, end, type) sang nhãn BIO chính xác tới từng byte."""
    enc = tok(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    seq_ids, labels = enc.sequence_ids(0), []
    for idx, (cs, ce) in enumerate(enc["offset_mapping"]):
        if seq_ids[idx] is None or cs == ce:
            labels.append(-100)
            continue
        hit = next(((s, e, t) for s, e, t in spans if ce > s and cs < e), None)
        labels.append(label2id["O" if hit is None else f"{'B' if cs <= hit[0] else 'I'}-{hit[2]}"])
    enc["labels"] = labels
    return enc

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
ner_net = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=7)
