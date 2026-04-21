import os
import cv2
import pandas as pd

# import your pipeline
from pipeline.stage1_preprocess import preprocess
from pipeline.stage2_segment import segment_leaf
from pipeline.stage3_features import extract_features


# -------------------------------
# STEP 1: BUILD LABEL MAPPING
# -------------------------------
def build_label_mapping(csv_path):
    df = pd.read_csv(csv_path)

    mapping = {}

    for _, row in df.iterrows():
        species = row["Scientific Name"]
        file_range = str(row["filename"])

        if "-" not in file_range:
            continue

        start, end = file_range.split("-")
        start, end = int(start), int(end)

        for i in range(start, end + 1):
            mapping[f"{i}.jpg"] = species

    return mapping


# -------------------------------
# STEP 2: RUN PIPELINE
# -------------------------------
def run(input_folder, label_csv):

    label_map = build_label_mapping(label_csv)

    results = []

    os.makedirs("data/masks", exist_ok=True)

    for file in os.listdir(input_folder):

        if not file.endswith(".jpg"):
            continue

        path = os.path.join(input_folder, file)

        img = cv2.imread(path)

        if img is None:
            print(f"Failed: {file}")
            continue

        # -----------------------
        # Preprocess
        # -----------------------
        img_proc = preprocess(img)

        # -----------------------
        # Segment (use your best version here)
        # -----------------------
        mask, contour = segment_leaf(img_proc, config={
            "segmentation": {
                "hsv_lower": [20, 30, 30],
                "hsv_upper": [95, 255, 255]
            }
        })

        if mask is None:
            print(f"Skipped: {file}")
            continue

        # -----------------------
        # Features
        # -----------------------
        feats = extract_features(img_proc, mask)

        if feats is None:
            continue

        # -----------------------
        # Attach label
        # -----------------------
        species = label_map.get(file)

        if species is None:
            print(f"No label: {file}")
            continue

        feats["filename"] = file
        feats["species"] = species

        # optional quality metrics
        feats["mask_quality"] = mask.sum() / (mask.size * 255)

        results.append(feats)

        print(f"Processed: {file}")

    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(results)
    df.to_csv("data/features.csv", index=False)

    print("\nDONE: data/features.csv created")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    run("data/raw", "flavia_labels.csv")