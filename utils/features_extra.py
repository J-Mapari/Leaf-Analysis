import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def fractal_dimension(mask):
    mask = mask > 0
    sizes, counts = [], []

    for size in [2,4,8,16,32]:
        S = cv2.resize(mask.astype(np.uint8), (size, size))
        counts.append(np.sum(S > 0))
        sizes.append(size)

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def color_features(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask == 255]

    return {
        "hue_mean": np.mean(pixels[:,0]),
        "hue_std": np.std(pixels[:,0]),
        "sat_mean": np.mean(pixels[:,1]),
        "val_mean": np.mean(pixels[:,2])
    }


def texture_features(gray, mask):
    roi = gray[mask==255]

    glcm = graycomatrix(roi.reshape(-1,1), [1], [0], levels=256)

    return {
        "contrast": graycoprops(glcm, 'contrast')[0,0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0,0]
    }