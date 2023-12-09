import cv2
import numpy as np
from segmentation.thresholding import thresholding


def min_sorted_contours(sorted_contours):
    return min(cv2.boundingRect(ctr)[1] for ctr in sorted_contours)


def find_contours(letter, sort):
    (contours, heirarchy) = cv2.findContours(letter.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[sort])
    return sorted_contours


def merge_letters(letters):
    
    def hconcat_resize_min(im_list):
        h_min = min(im.shape[0] for im in im_list)
        im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=cv2.INTER_CUBIC)
                          for im in im_list]
        return cv2.hconcat(im_list_resize)
    
    def is_eligible_for_merge(letter, sorted_contours):
        if len(sorted_contours) < 2:
            return False
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        y_next = min_sorted_contours(sorted_contours[1:])
        return y_next > y + h and y + h < letter.shape[0] // 2 and x + w == letter.shape[1] and w * h > 10

    unite_letters = []
    i = 0
    while i < len(letters):
        sorted_contours = find_contours(letters[i], 1)

        if is_eligible_for_merge(letters[i], sorted_contours) and i != len(letters) - 1:
            sorted_contours_next = find_contours(letters[i + 1], 1)

            if is_eligible_for_merge(letters[i + 1], sorted_contours_next) and cv2.boundingRect(sorted_contours_next[0])[0] == 0:
                unite_letters.append(hconcat_resize_min([letters[i], letters[i + 1]]))
                i += 2
                continue

        unite_letters.append(letters[i])
        i += 1

    return unite_letters


def letter_segmentation(word):
    dst = cv2.fastNlMeansDenoising(word, None, 12, 7, 21)
    thresh_img = thresholding(dst)

    kernel = np.ones((2, 1), np.uint8)
    dilated = cv2.dilate(thresh_img, kernel, iterations=1)    
    sorted_contours = find_contours(dilated, 0)

    top = thresh_img.shape[0] / 2
    bottom = thresh_img.shape[0] * 2 / 3
    min_area = 10

    letters = []
    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if y + h > top + 3 and y - 1 < bottom and w * h >= min_area:
            crop_x = max(x - 1, 0)
            letter = thresh_img[:y + h + 1, crop_x:x + w + 1]
            letters.append(letter)
            
    return merge_letters(letters)