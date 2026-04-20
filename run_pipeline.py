import os
import cv2
import yaml
import pandas as pd
import logging

from pipeline.stage1_preprocess import preprocess
from pipeline.stage2_segment import segment_leaf
from pipeline.stage3_features import extract_features

from utils.calibration import leaf_cropper
from utils.visualization import save_debug


# -------------------------------
# Logging setup
# -------------------------------
def setup_logging():
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/pipeline.log"),
            logging.StreamHandler()
        ]
    )


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run(input_folder):

    setup_logging()
    logging.info("Starting pipeline")

    config = load_config()
    results = []

    os.makedirs("data/masks", exist_ok=True)
    os.makedirs("data/debug", exist_ok=True)

    total = 0
    processed = 0
    skipped = 0
    crop_failed = 0

    for class_name in os.listdir(input_folder):

        class_path = os.path.join(input_folder, class_name)

        if not os.path.isdir(class_path):
            continue

        logging.info(f"Processing class: {class_name}")

        for file in os.listdir(class_path):

            total += 1
            path = os.path.join(class_path, file)

            img = cv2.imread(path)

            if img is None:
                logging.warning(f"Unreadable image: {file}")
                skipped += 1
                continue

            # --- Crop ---
            cropped = leaf_cropper(img)
            if cropped is None:
                logging.warning(f"Crop failed: {file}")
                crop_failed += 1
                continue

            # --- Preprocess ---
            img_proc = preprocess(cropped)

            # --- Segment ---
            mask, contour = segment_leaf(img_proc, config)

            if mask is None:
                logging.warning(f"Segmentation failed: {file}")
                skipped += 1
                continue

            # --- Save outputs ---
            mask_path = f"data/masks/{file}"
            debug_path = f"data/debug/{file}"

            cv2.imwrite(mask_path, mask)
            save_debug(cropped, contour, debug_path)

            # --- Features ---
            feats = extract_features(img_proc, mask)

            if feats is None:
                logging.warning(f"Feature extraction failed: {file}")
                skipped += 1
                continue

            feats["filename"] = file
            feats["label"] = class_name

            results.append(feats)
            processed += 1

            logging.info(f"Processed: {file}")

    # -------------------------------
    # Save CSV
    # -------------------------------
    df = pd.DataFrame(results)
    df.to_csv("data/features.csv", index=False)

    logging.info("Feature extraction complete.")

    # -------------------------------
    # Summary
    # -------------------------------
    logging.info("----- SUMMARY -----")
    logging.info(f"Total images: {total}")
    logging.info(f"Processed: {processed}")
    logging.info(f"Skipped: {skipped}")
    logging.info(f"Crop failed: {crop_failed}")

    # --- Train ML model (optional) ---
    # train_model("data/features.csv")


if __name__ == "__main__":
    run("data/raw")