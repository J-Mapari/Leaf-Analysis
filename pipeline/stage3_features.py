import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_features(img, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    # -----------------------
    # BASIC GEOMETRY
    # -----------------------
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return None

    x, y, w, h = cv2.boundingRect(cnt)

    extent = area / (w * h + 1e-6)

    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)

    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)

    # -----------------------
    # CONVEX HULL FEATURES
    # -----------------------
    hull = cv2.convexHull(cnt)

    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    hull_perimeter = cv2.arcLength(hull, True)
    convexity = hull_perimeter / (perimeter + 1e-6)

    # -----------------------
    # SIZE / SHAPE
    # -----------------------
    eq_diameter = np.sqrt((4 * area) / np.pi)

    elongation = (max(w, h) - min(w, h)) / (max(w, h) + min(w, h) + 1e-6)

    rectangularity = area / (w * h + 1e-6)

    # -----------------------
    # HU MOMENTS (VERY IMPORTANT)
    # -----------------------
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    # log scale (stabilises values)
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # -----------------------
    # COLOUR FEATURES
    # -----------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    pixels = hsv[mask == 255]

    if len(pixels) == 0:
        return None

    hue_mean = np.mean(pixels[:, 0])
    hue_std = np.std(pixels[:, 0])

    sat_mean = np.mean(pixels[:, 1])
    sat_std = np.std(pixels[:, 1])

    val_mean = np.mean(pixels[:, 2])
    val_std = np.std(pixels[:, 2])

    # -----------------------
    # TEXTURE FEATURES (GLCM)
    # -----------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roi = gray[mask == 255]

    if len(roi) < 10:
        contrast = 0
        homogeneity = 0
    else:
        # reshape safely
        roi = roi.reshape(-1, 1)

        glcm = graycomatrix(roi, [1], [0], levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # -----------------------
    # FEATURE DICT
    # -----------------------
    features = {
        "area": area,
        "perimeter": perimeter,
        "extent": extent,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "convexity": convexity,
        "equivalent_diameter": eq_diameter,
        "elongation": elongation,
        "rectangularity": rectangularity,

        # colour
        "hue_mean": hue_mean,
        "hue_std": hue_std,
        "sat_mean": sat_mean,
        "sat_std": sat_std,
        "val_mean": val_mean,
        "val_std": val_std,

        # texture
        "contrast": contrast,
        "homogeneity": homogeneity
    }

    # add Hu moments
    for i in range(7):
        features[f"hu_{i}"] = hu[i]

    return features