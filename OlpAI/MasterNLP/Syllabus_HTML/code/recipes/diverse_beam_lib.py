from transformers import AutoModelForSeq2SeqLM, GenerationConfig

# Sinh 16 ứng viên đa dạng bằng Diverse Beam Search có sẵn làm đầu vào cho bộ chấm điểm tương phản SimCLS:
div_cfg = GenerationConfig(
    num_beams=16, num_beam_groups=4, diversity_penalty=1.0,
    num_return_sequences=16, max_new_tokens=128, no_repeat_ngram_size=3,
)
cand_gen = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
