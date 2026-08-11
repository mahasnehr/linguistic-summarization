# Enhanced T5 hyperparameter tuning for multi-label emotion detection
import os, re, random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback, set_seed
)

SEED = 42
DATA_FILE = "preprocessed_data.csv"
MODEL_NAME = "t5-small"
OUTPUT_DIR = "./t5_tuned_results"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4

# Validation-driven hyperparameter search.
HYPERPARAMETER_CONFIGS = [
    dict(learning_rate=1e-5, batch_size=16, weight_decay=0.01,
         warmup_ratio=0.10, label_smoothing=0.05, gradient_accumulation_steps=1),
    dict(learning_rate=2e-5, batch_size=16, weight_decay=0.01,
         warmup_ratio=0.10, label_smoothing=0.05, gradient_accumulation_steps=1),
    dict(learning_rate=3e-5, batch_size=16, weight_decay=0.01,
         warmup_ratio=0.10, label_smoothing=0.05, gradient_accumulation_steps=1),
    dict(learning_rate=2e-5, batch_size=8, weight_decay=0.05,
         warmup_ratio=0.10, label_smoothing=0.10, gradient_accumulation_steps=2),
    dict(learning_rate=1e-5, batch_size=8, weight_decay=0.05,
         warmup_ratio=0.15, label_smoothing=0.10, gradient_accumulation_steps=2),
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# -------------------- Data --------------------
df = pd.read_csv(DATA_FILE)
if not {"text", "labels"}.issubset(df.columns):
    raise ValueError("CSV must contain 'text' and 'labels' columns.")

def normalize_labels(x):
    if isinstance(x, list):
        vals = x
    else:
        x = str(x).strip()
        if not x or x.lower() in {"nan", "none", "[]"}:
            return []
        x = re.sub(r"[\[\]'\" ]", "", x).replace(",", ";")
        vals = x.split(";")
    return sorted(set(str(v).strip().lower() for v in vals if str(v).strip()))

df["text"] = df["text"].fillna("").astype(str)
df["labels_list"] = df["labels"].apply(normalize_labels)
df["target_text"] = df["labels_list"].apply(lambda x: " ; ".join(x))
df = df[df["text"].str.strip().ne("")].reset_index(drop=True)

dataset = Dataset.from_pandas(df[["text", "target_text"]], preserve_index=False)

# 80/10/10 split.
split = dataset.train_test_split(test_size=0.20, seed=SEED)
train_ds = split["train"]
tmp_ds = split["test"]
split2 = tmp_ds.train_test_split(test_size=0.50, seed=SEED)
valid_ds, test_ds = split2["train"], split2["test"]

print(f"Train: {len(train_ds)} | Validation: {len(valid_ds)} | Test: {len(test_ds)}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    x = tokenizer(batch["text"], max_length=MAX_INPUT_LENGTH,
                  truncation=True, padding=False)
    y = tokenizer(text_target=batch["target_text"],
                  max_length=MAX_TARGET_LENGTH, truncation=True, padding=False)
    x["labels"] = y["input_ids"]
    return x

train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
valid_tok = valid_ds.map(tokenize, batched=True, remove_columns=valid_ds.column_names)
test_tok = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)

# -------------------- Metrics --------------------
def parse_labels(text):
    text = str(text).strip().lower()
    if not text:
        return []
    parts = re.split(r"\s*;\s*|\s*,\s*", text)
    return sorted(set(p.strip() for p in parts if p.strip()))

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    true_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

    y_true_l = [parse_labels(x) for x in true_text]
    y_pred_l = [parse_labels(x) for x in pred_text]

    mlb = MultiLabelBinarizer()
    mlb.fit(y_true_l + y_pred_l)
    y_true = mlb.transform(y_true_l)
    y_pred = mlb.transform(y_pred_l)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

def make_model():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.config.dropout_rate = 0.10
    return model

# -------------------- One tuning trial --------------------
def run_trial(config, trial):
    trial_dir = os.path.join(OUTPUT_DIR, f"trial_{trial}")
    os.makedirs(trial_dir, exist_ok=True)

    model = make_model()
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    args = Seq2SeqTrainingArguments(
        output_dir=trial_dir,
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        max_grad_norm=1.0,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        generation_num_beams=4,
        label_smoothing_factor=config["label_smoothing"],
        fp16=torch.cuda.is_available(),
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=valid_tok,
        tokenizer=tokenizer, data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
    )

    trainer.train()
    metrics = trainer.evaluate(metric_key_prefix="validation")
    result = {
        "trial": trial, **config,
        "validation_accuracy": metrics.get("validation_accuracy", 0),
        "validation_precision_micro": metrics.get("validation_precision_micro", 0),
        "validation_recall_micro": metrics.get("validation_recall_micro", 0),
        "validation_f1_micro": metrics.get("validation_f1_micro", 0),
    }
    trainer.save_model(os.path.join(trial_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(trial_dir, "best_model"))
    return result

# -------------------- Search --------------------
results = []
for i, config in enumerate(HYPERPARAMETER_CONFIGS, 1):
    print("\n" + "="*60)
    print("Trial", i, config)
    result = run_trial(config, i)
    results.append(result)
    print("Validation F1:", round(result["validation_f1_micro"], 4))

results_df = pd.DataFrame(results).sort_values("validation_f1_micro", ascending=False)
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "hyperparameter_search_results.csv"), index=False)

best = results_df.iloc[0]
best_trial = int(best["trial"])
best_path = os.path.join(OUTPUT_DIR, f"trial_{best_trial}", "best_model")

print("\nBest configuration:")
print(best.to_dict())

# -------------------- Final test evaluation --------------------
final_model = T5ForConditionalGeneration.from_pretrained(best_path)
final_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=final_model, padding="longest")

final_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "final"),
    per_device_eval_batch_size=int(best["batch_size"]),
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=4,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

final_trainer = Seq2SeqTrainer(
    model=final_model, args=final_args, eval_dataset=test_tok,
    tokenizer=tokenizer, data_collator=final_collator,
    compute_metrics=compute_metrics
)

test_metrics = final_trainer.evaluate(metric_key_prefix="test")
print("\nFINAL TEST RESULTS")
print(f"Accuracy:        {test_metrics.get('test_accuracy', 0):.4f}")
print(f"Precision micro: {test_metrics.get('test_precision_micro', 0):.4f}")
print(f"Recall micro:    {test_metrics.get('test_recall_micro', 0):.4f}")
print(f"F1 micro:        {test_metrics.get('test_f1_micro', 0):.4f}")

pd.DataFrame([test_metrics]).to_csv(
    os.path.join(OUTPUT_DIR, "final_test_metrics.csv"), index=False
)
