import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.morphology import skeletonize

def shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return {}
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt,True)
    compactness = (perimeter**2)/(area + 1e-6)
    return { 
        "area": area, 
        "perimeter": perimeter, 
        "compactness": compactness
    }

def colour_features(img, mask): 
    pixels = img[mask==255]
    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)

    return {
        "mean_r": mean[2],
        "mean_g": mean[1],
        "mean_b": mean[0],
        "std_r": std[2],
        "std_g": std[1],
        "std_b": std[0],
    }

def glcm_features(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = (gray/16).astype("uint8")
    glcm = graycomatrix(gray, [1], [0], 16, symmetric=True, normed=True)
    return {
        "contrast": graycoprops(glcm, 'contrast')[0,0],
        "homogeneity": graycoprops(glcm, 'homogeneity')[0,0],
        "energy": graycoprops(glcm, 'correlation')[0,0]
    }

def lbp_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8,1 method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins = np.arrange(0,11), range=(0,10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return {f"lbp_{i}": hist[i] for i in range(len(hist))}

def venation_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(thresh//255)
    return { 
        "vein_density": np.sum(skeleton)/(gray.size+1e-6)
    }

def extract_features(img, mask):
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    f = {}
    f.update(shape_features(mask))
    f.update(colour_features(masked_img, mask))
    f.update(glcm_features(masked_img))
    f.update(lbp_features(masked_img))
    f.update(venation_features(masked_img))
    return f

