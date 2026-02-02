import sys
import os
import yaml
import numpy as np
from pathlib import Path

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)

# -------------------------------------------------
# PATH FIX & Resolve roots
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now use full import path
from classifier.src.modeling.model_factory import build_model

# Classifier root for configs/data/artifacts
CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Load config
# -------------------------------------------------

def load_config():
    cfg_path = CLASSIFIER_ROOT / "configs" / "model.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Metrics
# -------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# -------------------------------------------------
# Dataset cleaning
# -------------------------------------------------

def clean_split(ds):
    # Rename label column if needed
    if "label_id" in ds.column_names:
        ds = ds.rename_column("label_id", "labels")

    # Only keep tensor-safe columns
    allowed = {"input_ids", "attention_mask", "labels"}

    if "token_type_ids" in ds.column_names:
        allowed.add("token_type_ids")

    drop_cols = [c for c in ds.column_names if c not in allowed]
    print("Dropping columns:", drop_cols)

    ds = ds.remove_columns(drop_cols)

    ds.set_format("torch")
    return ds


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    # -------------------------------------------------
    # Load tokenized datasets
    # -------------------------------------------------

    tokenized_dir = CLASSIFIER_ROOT / cfg["data"]["tokenized_path"]

    print(f"Loading tokenized datasets from: {tokenized_dir}")

    datasets = load_from_disk(tokenized_dir)

    train_ds = datasets["train"]
    val_ds = datasets["validation"]

    # -------------------------------------------------
    # Inspect label distribution
    # -------------------------------------------------

    if "label_id" in train_ds.column_names:
        labels_for_count = train_ds["label_id"]
    else:
        labels_for_count = train_ds["labels"]

    label_counts = np.bincount(labels_for_count)
    print("Train label distribution:", label_counts)

    # -------------------------------------------------
    # Clean splits
    # -------------------------------------------------

    train_ds = clean_split(train_ds)
    val_ds = clean_split(val_ds)

    # -------------------------------------------------
    # Tokenizer
    # -------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["pretrained_name"]
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = build_model(cfg)

    # -------------------------------------------------
    # Training arguments
    # -------------------------------------------------

    out_dir = CLASSIFIER_ROOT / cfg["training"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
        per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
        num_train_epochs=int(cfg["training"]["epochs"]),
        learning_rate=float(cfg["training"]["lr"]),
        weight_decay=cfg["training"].get("weight_decay", 0.0),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.0),
        logging_steps=cfg["training"].get("logging_steps", 50),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    # -------------------------------------------------
    # Trainer
    # -------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    print("🚀 Starting training...")
    trainer.train()

    # -------------------------------------------------
    # Save final model
    # -------------------------------------------------

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving final model to: {final_dir}")

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
