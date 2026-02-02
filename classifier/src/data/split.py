# src/data/split.py

import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# PATH FIX & Resolve roots
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Classifier root for data access
CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]


def split_and_save(df: pd.DataFrame, output_dir: str, 
                   test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42):
    """
    Split dataframe into train/val/test and save to CSV files.
    
    Args:
        df: Preprocessed dataframe
        output_dir: Directory to save split datasets
        test_size: Proportion for test set (default 0.2)
        val_size: Proportion for validation set (default 0.1)
        random_state: Random seed for reproducibility
    
    Returns:
        train, val, test dataframes
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=df["label_id"]
    )
    
    # Second split: separate validation from training
    val_proportion = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val["label_id"]
    )
    
    # Save to CSV
    train.to_csv(output_path / "train.csv", index=False)
    val.to_csv(output_path / "val.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    
    print(f"✓ Saved train.csv ({len(train)} rows)")
    print(f"✓ Saved val.csv ({len(val)} rows)")
    print(f"✓ Saved test.csv ({len(test)} rows)")
    
    return train, val, test


if __name__ == "__main__":
    from classifier.src.data.load_data import load_raw_dataset
    from classifier.src.data.preprocess import preprocess_dataframe
    
    data_dir = CLASSIFIER_ROOT / "data" / "raw" / "thought_dataset_10k_ultrasaiyan_diverse.xlsx"
    output_dir = CLASSIFIER_ROOT / "data" / "processed"
    
    df = load_raw_dataset(data_dir)
    df = preprocess_dataframe(df)
    train, val, test = split_and_save(df, output_dir)
    
    print("\nTrain distribution:")
    print(train["label"].value_counts())
