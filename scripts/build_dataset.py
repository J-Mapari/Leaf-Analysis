import os
import cv2
import pandas as pd

from pipeline.segment import segment_leaf
from pipeline.features import extract_features
from pipeline.metadata import parse_xml


def process(folder, output):

    rows = []
    files = os.listdir(folder)

    images = [f for f in files if f.lower().endswith(".jpg")]

    skipped_xml = 0
    skipped_meta = 0
    skipped_seg = 0

    for i, img_file in enumerate(images):

        base = os.path.splitext(img_file)[0]

        xml_path = os.path.join(folder, base + ".xml")
        img_path = os.path.join(folder, img_file)

        # -------------------------
        # CHECK XML EXISTS
        # -------------------------
        if not os.path.exists(xml_path):
            skipped_xml += 1
            continue

        # -------------------------
        # LOAD IMAGE
        # -------------------------
        img = cv2.imread(img_path)
        if img is None:
            continue

        # -------------------------
        # SEGMENT
        # -------------------------
        crop, mask = segment_leaf(img)

        if crop is None or mask is None:
            skipped_seg += 1
            continue

        # -------------------------
        # FEATURES
        # -------------------------
        feats = extract_features(crop, mask)

        # -------------------------
        # METADATA
        # -------------------------
        meta = parse_xml(xml_path)

        # CRITICAL CHECK
        if meta is None:
            skipped_meta += 1
            continue
        # -------------------------
        # BUILD ROW
        # -------------------------
        row = {
            "image": img_file,   # ← FIXED: ensure image column exists
            **meta,
            **feats
        }

        rows.append(row)

        if i % 200 == 0:
            print(f"{i}/{len(images)}")

    # -------------------------
    # SAVE
    # -------------------------
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

    print("\nSaved:", output)
    print("Total:", len(images))
    print("Kept:", len(df))
    print("Skipped (no XML):", skipped_xml)
    print("Skipped (bad meta):", skipped_meta)
    print("Skipped (seg fail):", skipped_seg)


if __name__ == "__main__":
    process("data/train", "outputs/train.csv")  