import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# Paths
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
CLINICAL_CSV = PROJECT_ROOT / "data" / "clinical" / "anemia.csv"

OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "fusion_data.csv"

# ======================================================
# Load datasets
# ======================================================

image_df = pd.read_csv(IMAGE_CSV)
clinical_df = pd.read_csv(CLINICAL_CSV)

print("Image Samples    :", len(image_df))
print("Clinical Samples :", len(clinical_df))

# ======================================================
# Repeat clinical rows if needed
# ======================================================

if len(clinical_df) < len(image_df):

    repeats = int(np.ceil(len(image_df) / len(clinical_df)))

    clinical_df = pd.concat(
        [clinical_df] * repeats,
        ignore_index=True
    )

clinical_df = clinical_df.iloc[:len(image_df)].reset_index(drop=True)

# ======================================================
# Merge
# ======================================================

fusion_df = image_df.copy()

fusion_df["Gender"] = clinical_df["Gender"]
fusion_df["Hemoglobin"] = clinical_df["Hemoglobin"]
fusion_df["MCH"] = clinical_df["MCH"]
fusion_df["MCHC"] = clinical_df["MCHC"]
fusion_df["MCV"] = clinical_df["MCV"]

# Geo feature
fusion_df["Geo"] = fusion_df["country"].map({
    "India": 0,
    "Italy": 1
})

fusion_df.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print("Fusion Dataset Created Successfully")
print("=" * 60)
print(fusion_df.head())

print("\nSaved to:")
print(OUTPUT_CSV)