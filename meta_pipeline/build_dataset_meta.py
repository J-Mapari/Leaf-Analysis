import os
import cv2
import pandas as pd

from pipeline.segment import segment_leaf
from pipeline.features import extract_features
from pipeline.metadata import parse_xml
from meta_pipeline.metadata_features import extract_metadata_features

INPUT = "data/train"
OUTPUT = "outputs_meta/train_meta.csv"

rows = []

files = os.listdir(INPUT)
images = [f for f in files if f.endswith(".jpg")]

for i, img_file in enumerate(images):

    base = img_file.replace(".jpg", "")
    xml_path = os.path.join(INPUT, base + ".xml")
    img_path = os.path.join(INPUT, img_file)

    if not os.path.exists(xml_path):
        continue

    img = cv2.imread(img_path)
    crop, mask = segment_leaf(img)

    if crop is None:
        continue

    feats = extract_features(crop, mask)
    meta = parse_xml(xml_path)
    meta_feats = extract_metadata_features(meta)

    row = {
        "image": img_file,
        "class_id": meta.get("ClassId", None),
        **feats,
        **meta_feats
    }

    rows.append(row)

    if i % 200 == 0:
        print(i)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)