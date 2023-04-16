import cv2
import numpy as np
from imutils import rotate


def horizontal_projection(sobel_img):
    sum_cols = []
    rows, cols = sobel_img.shape
    for row in range(rows-1):
        sum_cols.append(np.sum(sobel_img[row,:]))
    return sum_cols


def rotate_image(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel_img = np.sqrt(gx * gx + gy * gy)
    sobel_img_inv = 255 - sobel_img
    predicted_angle = 0
    highest_hp = 0
    for index, angle in enumerate(range(-5, 5)):
        hp = horizontal_projection(rotate(sobel_img_inv, angle))
        median_hp = np.median(hp)
        if highest_hp < median_hp:
            predicted_angle = angle
            highest_hp = median_hp
    rows = img.shape[0]
    cols = img.shape[1]
    img_center = (cols / 2, rows / 2)
    matrix = cv2.getRotationMatrix2D(img_center, predicted_angle, 1)
    rotate_img = cv2.warpAffine(img, matrix, (cols, rows), borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
    return rotate_img, predicted_angle


def skew_correction(path_to_file):
    img = cv2.cvtColor(cv2.imread(path_to_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    rotate_img, angle = rotate_image(img)
    return rotate_img
