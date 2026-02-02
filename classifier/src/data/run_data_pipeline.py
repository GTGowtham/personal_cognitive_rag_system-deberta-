# src/data/run_data_pipeline.py

import sys
from pathlib import Path

# -------------------------------------------------
# PATH FIX & Resolve roots
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 1_THOUGHTS_CLASSIFIER
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Classifier root for data access
CLASSIFIER_ROOT = Path(__file__).resolve().parents[2]

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from classifier.src.data.load_data import load_raw_dataset
from classifier.src.data.preprocess import preprocess_dataframe
from classifier.src.data.split import split_and_save

RAW_PATH = CLASSIFIER_ROOT / "data" / "raw" / "thoughts.xlsx"
OUT_DIR = CLASSIFIER_ROOT / "data" / "processed"

print("=" * 50)
print("STARTING DATA PIPELINE")
print("=" * 50)

# Step 1: Load raw data
print("\n[1/3] Loading raw dataset...")
df = load_raw_dataset(RAW_PATH)
print(f"✓ Loaded {len(df)} rows")

# Step 2: Preprocess
print("\n[2/3] Preprocessing...")
df = preprocess_dataframe(df)
print(f"✓ Preprocessed {len(df)} rows")

# Step 3: Split and save
print("\n[3/3] Splitting and saving...")
train, val, test = split_and_save(df, OUT_DIR)

# Display results
print("\n" + "=" * 50)
print("PIPELINE COMPLETE")
print("=" * 50)

print("\n📊 Train distribution:")
print(train["label"].value_counts())

print("\n📊 Validation distribution:")
print(val["label"].value_counts())

print("\n📊 Test distribution:")
print(test["label"].value_counts())

print("\n✅ All files saved to:", OUT_DIR)
