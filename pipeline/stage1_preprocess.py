import cv2

def preprocess(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    image = cv2.merge((l,a,b))
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    image = cv2.GaussianBlur(image, (5,5), 0)
    return image
