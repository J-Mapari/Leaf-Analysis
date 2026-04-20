import cv2
import numpy as np


def segment_leaf(img, config):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lower = np.array(config["segmentation"]["hsv_lower"])
    upper = np.array(config["segmentation"]["hsv_upper"])

    color_mask = cv2.inRange(hsv, lower, upper)

    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    mask = cv2.bitwise_or(color_mask, edges)

    kernel = np.ones((7,7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels <= 1:
        return None, None

    largest_label = 1 + np.argmax([
        np.sum(labels == i) for i in range(1, num_labels)
    ])

    mask = np.uint8(labels == largest_label) * 255

    # --- Contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)

    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest], -1, 255, -1)

    return final_mask, largest