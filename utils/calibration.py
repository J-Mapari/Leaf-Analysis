import cv2 
import numpy as np 
import math 

def is_valid_leaf(contour, image_area): 
    area = cv2.contourArea(contour)
    if area < 0.01 * image_area: 
        return False
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h) 

    if aspect_ratio < 0.2 or aspect_ratio > 5: 
        return False
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False
    
    solidity = area/ hull_area
    if solidity < 0.6: 
        return False

    rect_area = w*h
    extent = area / rect_area
    if extent < 0.3: 
        return False
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: 
        return False 
    
    circularity = 4* math.pi  * (area / (perimeter * perimeter))
    if circularity < 0.1: 
        return False
    
    return True

def leaf_cropper(img): 

    image_area = img.shape[0] * img.shape[1]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([20,12,20])
    upper_green = np.array([95,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = [c for c in contours if is_valid_leaf(c, image_area)]

    if not valid: 
        return None 
    
    leaf = max(valid, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(leaf)

    margin = 30 

    return img[
        max(0, y-margin):min(img.shape[0], y+h+margin), 
        max(0,x-margin):min(img.shape[1], x+w+margin)
    ]
