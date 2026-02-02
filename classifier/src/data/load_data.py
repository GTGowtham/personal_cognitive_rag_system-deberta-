# src/data/load_data.py

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

# Define paths
data_dir = CLASSIFIER_ROOT / "data" / "raw" / "thoughts.xlsx"

required_columns = {"thought_text", "label"}


def load_raw_dataset(path):
    """Load the raw dataset from Excel file."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file {path} does not exist.")
    
    df = pd.read_excel(path)
    
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("The dataset is empty.")
    
    return df


if __name__ == "__main__":
    df = load_raw_dataset(data_dir)
    print(f"Loaded {len(df)} rows")
    print(df.head())
