from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
from tqdm.auto import tqdm

# ====== 1. Load model & tokenizer ======
model_dir = "/ViT5"  # mô hình ViT5 gốc
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
model.config.use_cache = False
model.gradient_checkpointing_enable()
# ====== 2. Load dataset từ CSV ======
data_files = {"train": "/data/Parallel-Corpus-Vie-VSL/Corpus-Vie-VSL-10K.csv"}
dataset = load_dataset("csv", data_files=data_files)

# ====== 3. Chuẩn bị hàm tokenize ======
def preprocess_function(examples):
    inputs = ["sign-grammar: " + ex for ex in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=64, truncation=True)
    labels = tokenizer(text_target=examples["output"], max_length=64, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ====== 4. Chia train / eval ======
tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# ====== 5. Data collator ======
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ====== 6. Thiết lập tham số huấn luyện ======
training_args = Seq2SeqTrainingArguments(
    output_dir="./vit5_large_sign_grammar",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=10,
    predict_with_generate=False,
    bf16=True,
    gradient_accumulation_steps=4,
    dataloader_num_workers=20,
    logging_dir="./logs",
    logging_steps=50,
    optim="adafactor",
    save_strategy="epoch"
)


# ====== 7. Trainer ======
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# ====== 8. Train ======
trainer.train()

# ====== 9. Lưu mô hình fine-tune ======
trainer.save_model("./vit5_sign_grammar_finetuned_10_epochs")
tokenizer.save_pretrained("./vit5_sign_grammar_finetuned_10_epochs")