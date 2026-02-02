import sys
import yaml
import numpy as np
from pathlib import Path
import pandas as pd

from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# -------------------------------------------------
# PATH FIX & Resolve roots
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
# Dataset cleaning (same logic as training)
# -------------------------------------------------

def clean_split(ds):
    if "label_id" in ds.column_names:
        ds = ds.rename_column("label_id", "labels")

    allowed = {"input_ids", "attention_mask", "labels"}
    if "token_type_ids" in ds.column_names:
        allowed.add("token_type_ids")

    drop_cols = [c for c in ds.column_names if c not in allowed]
    ds = ds.remove_columns(drop_cols)

    ds.set_format("torch")
    return ds


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------------------------
    # Load tokenized test set
    # -------------------------------------------------

    tokenized_dir = CLASSIFIER_ROOT / cfg["data"]["tokenized_path"]
    datasets = load_from_disk(tokenized_dir)

    test_ds = datasets["test"]
    test_ds = clean_split(test_ds)

    # -------------------------------------------------
    # Load trained model + tokenizer
    # -------------------------------------------------

    model_dir = CLASSIFIER_ROOT / cfg["training"]["output_dir"] / "final"

    print("Loading model from:", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.to(device)
    model.eval()

    # -------------------------------------------------
    # Run predictions
    # -------------------------------------------------

    all_preds = []
    all_labels = []

    loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=16,
    )

    with torch.no_grad():
        for batch in loader:

            labels = batch.pop("labels").to(device)

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    acc = accuracy_score(all_labels, all_preds)

    print("\n========== TEST RESULTS ==========")
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # -------------------------------------------------
    # Save misclassified examples
    # -------------------------------------------------

    errors_mask = all_preds != all_labels

    errors_df = pd.DataFrame(
        {
            "true_label": all_labels[errors_mask],
            "pred_label": all_preds[errors_mask],
        }
    )

    out_dir = CLASSIFIER_ROOT / "artifacts" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    errors_path = out_dir / "test_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    print("\nSaved misclassified samples to:", errors_path)

    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
