from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# Paths
# ---------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

metadata = pd.read_csv(DATA_DIR / "image_metadata.csv")

# ---------------------------------------------
# Train (70%)
# Validation (15%)
# Test (15%)
# ---------------------------------------------

train_df, temp_df = train_test_split(
    metadata,
    test_size=0.30,
    stratify=metadata["label"],
    random_state=42
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# ---------------------------------------------
# Save
# ---------------------------------------------

train_df.to_csv(DATA_DIR / "train.csv", index=False)

valid_df.to_csv(DATA_DIR / "valid.csv", index=False)

test_df.to_csv(DATA_DIR / "test.csv", index=False)

# ---------------------------------------------
# Summary
# ---------------------------------------------

print("=" * 60)
print("Dataset Split Complete")
print("=" * 60)

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(valid_df)}")
print(f"Test       : {len(test_df)}")

print("=" * 60)

print("\nTrain Distribution")
print(train_df["label"].value_counts())

print("\nValidation Distribution")
print(valid_df["label"].value_counts())

print("\nTest Distribution")
print(test_df["label"].value_counts())