import cv2

def save_debug(img, contour, path):

    debug = img.copy()
    cv2.drawContours(debug, [contour], -1, (0,0,255), 2)

    cv2.imwrite(path, debug)