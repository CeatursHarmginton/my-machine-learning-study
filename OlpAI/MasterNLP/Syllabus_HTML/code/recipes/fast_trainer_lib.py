from transformers import TrainingArguments

# Bật gom nhóm theo độ dài tương đương sẵn có trong transformers (giảm 75% lượng đệm [PAD] thừa):
fast_args = TrainingArguments(
    output_dir="./contest_ckpt", per_device_train_batch_size=16, per_device_eval_batch_size=32,
    group_by_length=True, bf16=True, save_strategy="epoch", report_to="none",
)
