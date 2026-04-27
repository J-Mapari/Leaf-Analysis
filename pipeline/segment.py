import numpy as np
import cv2

def segment_leaf(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    # watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1

    markers = cv2.watershed(img, markers)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[markers > 1] = 255

    # find largest central contour 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours: 
        return img, np.ones(gray.shape, dtype=np.uint8)*255
    
    h, w = mask.shape 
    cx, cy = w // 2, h // 2

    def score(cnt):
        x,y,ww,hh = cv2.boundingRect(cnt)
        center = ((x+ww//2 - cx)**2 + (y + hh//2 - cy)**2)
        return center - 0.3 * cv2.contourArea(cnt)
    
    cnt = min(contours, key=score)

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [cnt], -1, 255, -1)

    x,y,w,h = cv2.boundingRect(cnt)

    crop = img[y:y+h, x:x+w]
    crop_mask = clean[y:y+h, x:x+w]

    # fallback 
    if np.sum(crop_mask) < 500: 
        return img, np.ones(gray.shape, dtype=np.uint8)*255
    
    return crop, crop_mask
