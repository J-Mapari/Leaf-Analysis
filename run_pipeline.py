import os
import cv2
import yaml
import pandas as pd

from pipeline.stage1_preprocess import preprocess
from pipeline.stage2_segment import segment_leaf
from pipeline.stage3_features import extract_features
from pipeline.stage4_model import train_model

from utils.calibration import leaf_cropper
from utils.visualization import save_debug


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def run(input_folder):

    config = load_config()
    results = []

    os.makedirs("data/masks", exist_ok=True)
    os.makedirs("data/debug", exist_ok=True)

    for class_name in os.listdir(input_folder):

        class_path = os.path.join(input_folder, class_name)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):

            path = os.path.join(class_path, file)
            img = cv2.imread(path)

            if img is None:
                continue

            # -------------------------------
            # Preprocess
            # -------------------------------
            img_proc = preprocess(img)

            # -------------------------------
            # Segment
            # -------------------------------
            mask, contour = segment_leaf(img_proc, config)

            if mask is None:
                print(f"Skipped: {file}")
                continue

            # -------------------------------
            # Save outputs
            # -------------------------------
            cv2.imwrite(f"data/masks/{file}", mask)
            save_debug(img, contour, f"data/debug/{file}")

            # -------------------------------
            # Features
            # -------------------------------
            feats = extract_features(img_proc, mask)
            feats["filename"] = file
            feats["label"] = class_name

            results.append(feats)

            print(f"Processed: {file}")

    # -------------------------------
    # Save CSV
    # -------------------------------
    df = pd.DataFrame(results)
    df.to_csv("data/features.csv", index=False)

    print("\nFeature extraction complete.")

    # -------------------------------
    # Train ML model
    # -------------------------------
    train_model("data/features.csv")


if __name__ == "__main__":
    run("data/raw")