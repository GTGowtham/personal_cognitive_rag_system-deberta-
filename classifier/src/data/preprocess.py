# src/data/preprocess.py

import sys
import pandas as pd
from pathlib import Path

# -------------------------------------------------
# PATH FIX & Resolve roots
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Classifier root for data access
CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]


LABEL_MAP = {
    "Noise": 0,
    "Emotion": 1,
    "Curiosity": 2,
    "Problem": 3,
    "Idea": 4,
}


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataframe by cleaning text and mapping labels."""
    df = df.copy()

    # trim whitespace
    df["thought_text"] = df["thought_text"].astype(str).str.strip()

    # normalize spaces
    df["thought_text"] = df["thought_text"].str.replace(r"\s+", " ", regex=True)

    # map labels
    if not set(df["label"]).issubset(LABEL_MAP.keys()):
        bad = set(df["label"]) - set(LABEL_MAP.keys())
        raise ValueError(f"Unknown labels detected: {bad}")

    df["label_id"] = df["label"].map(LABEL_MAP)

    # drop empty texts after cleaning
    df = df[df["thought_text"].str.len() > 0]

    return df


if __name__ == "__main__":
    from classifier.src.data.load_data import load_raw_dataset
    
    data_dir = CLASSIFIER_ROOT / "data" / "raw" / "thought_dataset_10k_ultrasaiyan_diverse.xlsx"
    
    df = load_raw_dataset(data_dir)
    df = preprocess_dataframe(df)
    print(f"Preprocessed {len(df)} rows")
    print(df.head())
