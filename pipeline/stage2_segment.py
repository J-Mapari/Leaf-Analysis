import cv2
import numpy as np

def segment_leaf(img, config):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array(config["segmentation"]["hsv_lower"])
    upper = np.array(config["segmentation"]["hsv_upper"])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)

    final_mask = np.zeros_like(mask)
    cv2.drawContours(final_mask, [largest], -1, 255, -1)

    return final_mask, largest