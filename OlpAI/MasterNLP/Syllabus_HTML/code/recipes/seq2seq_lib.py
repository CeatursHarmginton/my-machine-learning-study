from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

# Sinh chuỗi Seq2Seq (BART / Flan-T5 / ViT5) tích hợp sẵn KV-Cache, phạt độ dài & chặn lặp 3-gram:
s2s_tok = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
s2s_net = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
gen_cfg = GenerationConfig(
    max_new_tokens=140, min_new_tokens=35, num_beams=4,
    length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True, use_cache=True,
)
