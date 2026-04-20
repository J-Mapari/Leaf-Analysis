import cv2
import numpy as np
from skimage.morphology import skeletonize


def extract_features(img, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x, y, w, h = cv2.boundingRect(cnt)

    # --- Basic ---
    extent = area / (w*h + 1e-6)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)

    # --- Convex hull ---
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    hull_perimeter = cv2.arcLength(hull, True)
    convexity = hull_perimeter / (perimeter + 1e-6)

    # --- Equivalent diameter ---
    eq_diameter = np.sqrt((4 * area) / np.pi)

    # --- Aspect ratio (rotation invariant) ---
    rect = cv2.minAreaRect(cnt)
    wr, hr = rect[1]

    if wr > 0 and hr > 0:
        aspect_ratio = max(wr, hr) / min(wr, hr)
    else:
        aspect_ratio = 0

    # --- Eccentricity ---
    eccentricity = 0
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (_, _), (MA, ma), _ = ellipse
        if ma != 0:
            eccentricity = np.sqrt(1 - (MA / ma) ** 2)

    # =========================
    # 🌿 VENATION FEATURES
    # =========================

    # skeletonize requires 0/1
    binary = mask // 255

    skeleton = skeletonize(binary).astype(np.uint8)

    skeleton_pixels = np.sum(skeleton)

    # vein density
    vein_density = skeleton_pixels / (area + 1e-6)

    # branch points
    kernel = np.array([
        [1,1,1],
        [1,10,1],
        [1,1,1]
    ])

    neighbors = cv2.filter2D(skeleton, -1, kernel)
    branch_points = np.sum((neighbors >= 13) & (skeleton == 1))

    # endpoints
    endpoints = np.sum((neighbors == 11) & (skeleton == 1))

    # =========================

    features = {
        "area": area,
        "perimeter": perimeter,
        "extent": extent,
        "circularity": circularity,
        "solidity": solidity,
        "convexity": convexity,
        "equivalent_diameter": eq_diameter,
        "aspect_ratio": aspect_ratio,
        "eccentricity": eccentricity,

        # venation
        "vein_density": vein_density,
        "skeleton_length": skeleton_pixels,
        "branch_points": branch_points,
        "endpoints": endpoints
    }

    # --- Hu moments ---
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    for i in range(7):
        features[f"hu_{i}"] = hu[i]

    return features