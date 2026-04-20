import cv2
import numpy as np

def extract_features(img, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    x,y,w,h = cv2.boundingRect(cnt)

    aspect_ratio = h / w if w != 0 else 0
    circularity = (4*np.pi*area)/(perimeter**2 + 1e-6)

    hull = cv2.convexHull(cnt)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity
    }