from pathlib import Path
import pandas as pd
from PIL import Image
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = PROJECT_ROOT / "data" / "eye_image"

rows = []

label_map = {
    "India": 1,
    "Italy": 0
}

for country in ["India", "Italy"]:

    country_dir = IMAGE_ROOT / country

    for patient_folder in sorted(country_dir.iterdir()):

        if not patient_folder.is_dir():
            continue

        for img_path in patient_folder.iterdir():

            if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            # ---------- Check if image is readable ----------
            valid = True

            try:
                Image.open(img_path).verify()
            except:
                img = cv2.imread(str(img_path))
                if img is None:
                    valid = False

            if not valid:
                continue

            relative_path = img_path.relative_to(IMAGE_ROOT)

            rows.append(
                {
                    "image_path": str(relative_path),
                    "label": label_map[country],
                    "country": country,
                    "patient_folder": patient_folder.name,
                    "filename": img_path.name,
                }
            )

metadata = pd.DataFrame(rows)

metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

save_path = PROJECT_ROOT / "data" / "processed" / "image_metadata.csv"

metadata.to_csv(save_path, index=False)

print("=" * 60)
print("Metadata Created Successfully")
print("=" * 60)
print("Valid Images :", len(metadata))
print("Saved :", save_path)
print("=" * 60)
print(metadata.head())