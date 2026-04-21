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
# DEBUG SAVE
# -------------------------------
def save_debug(img, mask, contour, filename):
    os.makedirs("data/debug", exist_ok=True)

    debug = img.copy()

    if contour is not None:
        cv2.drawContours(debug, [contour], -1, (0, 0, 255), 2)

    # overlay mask
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(debug, 0.7, mask_colored, 0.3, 0)

    cv2.imwrite(f"data/debug/{filename}", overlay)


# -------------------------------
# STEP 2: RUN PIPELINE
# -------------------------------
def run(input_folder, label_csv):

    label_map = build_label_mapping(label_csv)

    results = []

    os.makedirs("data/masks", exist_ok=True)
    os.makedirs("data/debug", exist_ok=True)
    os.makedirs("data/debug_bad", exist_ok=True)

    total = 0
    processed = 0
    skipped = 0
    failed = 0

    for file in os.listdir(input_folder):

        if not file.endswith(".jpg"):
            continue

        total += 1

        path = os.path.join(input_folder, file)

        img = cv2.imread(path)

        if img is None:
            print(f"[FAIL LOAD] {file}")
            failed += 1
            continue

        # -----------------------
        # Preprocess
        # -----------------------
        img_proc = preprocess(img)

        # -----------------------
        # Segment
        # -----------------------
        mask, contour = segment_leaf(img_proc, config={
            "segmentation": {
                "hsv_lower": [20, 30, 30],
                "hsv_upper": [95, 255, 255]
            }
        })

        if mask is None:
            print(f"[SKIP SEGMENT] {file}")
            skipped += 1
            continue

        # -----------------------
        # Save mask
        # -----------------------
        cv2.imwrite(f"data/masks/{file}", mask)

        # -----------------------
        # Features
        # -----------------------
        feats = extract_features(img_proc, mask)

        if feats is None:
            print(f"[SKIP FEATURES] {file}")
            skipped += 1
            continue

        # -----------------------
        # Label
        # -----------------------
        species = label_map.get(file)

        if species is None:
            print(f"[NO LABEL] {file}")
            skipped += 1
            continue

        # -----------------------
        # Quality metric
        # -----------------------
        mask_quality = mask.sum() / (mask.size * 255)

        feats["filename"] = file
        feats["species"] = species
        feats["mask_quality"] = mask_quality

        results.append(feats)

        # -----------------------
        # Save debug images
        # -----------------------
        save_debug(img, mask, contour, file)

        # Save bad masks separately (VERY useful)
        if mask_quality < 0.05:
            cv2.imwrite(f"data/debug_bad/{file}", mask)

        processed += 1

        print(f"[OK] {file} | mask_quality={mask_quality:.3f}")

    # -------------------------------
    # SAVE CSV
    # -------------------------------
    df = pd.DataFrame(results)
    df.to_csv("data/features.csv", index=False)

    # -------------------------------
    # SUMMARY
    # -------------------------------
    print("\n===== SUMMARY =====")
    print(f"Total:     {total}")
    print(f"Processed: {processed}")
    print(f"Skipped:   {skipped}")
    print(f"Failed:    {failed}")

    print("\nDONE: data/features.csv created")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    run("data/raw", "flavia_labels.csv")