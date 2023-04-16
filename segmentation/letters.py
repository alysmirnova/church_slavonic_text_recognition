import cv2
import numpy as np
from segmentation.thresholding import thresholding


def hconcat_resize_min(im_list):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min),
                                 interpolation=cv2.INTER_CUBIC)
                      for im in im_list]
    print('*', im_list_resize[0].shape, im_list_resize[1].shape)
    return cv2.hconcat(im_list_resize)


def min_sorted_contours(sorted_contours):
    min_y = 100
    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        if y < min_y:
            min_y = y
    return min_y


def find_contours(letter, sort):
    (contours, heirarchy) = cv2.findContours(letter.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[sort])
    return sorted_contours


def merge_letters(letters):
    i = 0
    unite_letters = []
    while i < len(letters):
        sorted_contours = find_contours(letters[i], 1)
        if len(sorted_contours) >= 2 and i != len(letters)-1:
            x1, y1, w1, h1 = cv2.boundingRect(sorted_contours[0])
            y2 = min_sorted_contours(sorted_contours[1:])
            if y2 > y1+h1 and y1+h1 < letters[i].shape[0]//2 and x1+w1 == letters[i].shape[1] and \
                    ((x1+w1)-x1)*((y1+h1)-y1) > 10:
                sorted_contours1 = find_contours(letters[i+1], 1)
                if len(sorted_contours1) >= 2:
                    x3, y3, w3, h3 = cv2.boundingRect(sorted_contours1[0])
                    y4 = min_sorted_contours(sorted_contours1[1:])
                    if y4 > y3+h3 and y3+h3 < letters[i].shape[0]//2 and x3 == 0 and \
                            ((x3+w3)-x3)*((y3+h3)-y3) > 10:
                        a = hconcat_resize_min([letters[i], letters[i+1]])
                        unite_letters.append(a)
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
    divider_indexes = []

    word_copy = word.copy()
    top = word_copy.shape[0] / 2
    bottom = word_copy.shape[0] / 3 * 2

    letters = []
    for ctr in sorted_contours:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(word_copy, (x, y), (x + w, y + h), (40, 100, 250), 1)
        if y + h > top + 3 and y - 1 < bottom:
            if ((x + w) - x) * ((y + h) - y) >= 10:
                if x != 0:
                    letter = thresh_img[:y + h + 1, x - 1:x + w + 1]
                    divider_indexes.append(x - 1)
                else:
                    letter = thresh_img[:y + h + 1, x:x + w + 1]
                    divider_indexes.append(x)
                letters.append(letter)
    return merge_letters(letters)
