import sys
from pathlib import Path
import pandas as pd
import yaml

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


# -------------------------------------------------
# PATH FIX & Resolve roots
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Classifier root for configs/data
CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_split(csv_path: Path) -> Dataset:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return Dataset.from_pandas(df)


def build_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset: Dataset, tokenizer, text_col: str, max_length: int):
    return dataset.map(
        lambda x: tokenizer(
            x[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
    )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    # ----- Load config -----
    cfg = load_config(CLASSIFIER_ROOT / "configs" / "model.yaml")

    model_name = cfg["model"]["pretrained_name"]
    max_length = cfg["tokenization"]["max_length"]
    text_col = cfg["data"]["text_column"]
    output_path = CLASSIFIER_ROOT / cfg["data"]["tokenized_path"]

    print("Model:", model_name)
    print("Max length:", max_length)
    print("Saving tokenized datasets to:", output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # ----- Load tokenizer -----
    tokenizer = build_tokenizer(model_name)

    # ----- Load CSV splits -----
    train_ds = load_split(CLASSIFIER_ROOT / "data" / "processed" / "train.csv")
    val_ds = load_split(CLASSIFIER_ROOT / "data" / "processed" / "val.csv")
    test_ds = load_split(CLASSIFIER_ROOT / "data" / "processed" / "test.csv")

    # ----- Tokenize -----
    train_tok = tokenize_dataset(train_ds, tokenizer, text_col, max_length)
    val_tok = tokenize_dataset(val_ds, tokenizer, text_col, max_length)
    test_tok = tokenize_dataset(test_ds, tokenizer, text_col, max_length)

    # ----- Build DatasetDict -----
    tokenized = DatasetDict(
        {
            "train": train_tok,
            "validation": val_tok,
            "test": test_tok,
        }
    )

    # ----- Save datasets to disk -----
    tokenized.save_to_disk(str(output_path))

    # ----- Save tokenizer too -----
    tokenizer.save_pretrained(str(output_path / "tokenizer"))

    # ----- Final sanity -----
    print("\nTokenized dataset saved successfully.")
    print(tokenized)

    print("\nSample example:")
    print(
        {
            "input_ids": tokenized["train"][0]["input_ids"],
            "attention_mask": tokenized["train"][0]["attention_mask"],
            "label_id": tokenized["train"][0]["label_id"],
        }
    )

    print("\nDONE — tokenization artifacts written to disk.")


if __name__ == "__main__":
    main()
