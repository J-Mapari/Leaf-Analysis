import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd

from pipeline.segment import segment_leaf
from pipeline.features import extract_features

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🌿 Leaf Classification Demo")

st.warning("⚠️ Model performs well on scanned images but poorly on natural photographs.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image")
    st.image(img_rgb, use_column_width=True)

    # -------------------------
    # SEGMENT
    # -------------------------
    crop, mask = segment_leaf(img)

    if crop is None:
        st.error("Segmentation failed")
    else:

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # overlay
        overlay = crop.copy()
        overlay[mask == 0] = (0,0,0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        st.subheader("Segmentation Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(crop_rgb, caption="Crop")

        with col2:
            st.image(mask, caption="Mask", clamp=True)

        with col3:
            st.image(overlay_rgb, caption="Overlay")

        # -------------------------
        # FEATURES
        # -------------------------
        feats = extract_features(crop, mask)
        df = pd.DataFrame([feats])

        # -------------------------
        # PREDICT
        # -------------------------
        pred = model.predict(df)[0]

        st.subheader("Prediction")
        st.success(f"{pred}")

        # -------------------------
        # OPTIONAL: SHOW FEATURES
        # -------------------------
        if st.checkbox("Show extracted features"):
            st.dataframe(df)