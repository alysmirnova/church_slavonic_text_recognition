import cv2
import numpy as np
from segmentation.thresholding import thresholding


def cropp_line(line_image):
    thresh = thresholding(line_image)
    binary = line_image > thresh
    vertical_projection = np.sum(binary, axis=0)
    height = line_image.shape[0]
    index = 0
    while vertical_projection[index] == height:
        index += 1
    if index > 2:
        index -= 2
    line_image = line_image[:, index:]

    thresh = thresholding(line_image)
    binary = line_image > thresh
    vertical_projection = np.sum(binary, axis=0)

    index = line_image.shape[1] - 1
    while vertical_projection[index] == height:
        index -= 1
    if index < line_image.shape[1] - 1:
        index += 2

    line_image = line_image[:, :index]
    return line_image


def word_segmentation(line_image):
    line = cropp_line(line_image)
    dst = cv2.fastNlMeansDenoising(line, None, 12, 7, 21)
    thresh = thresholding(dst)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    binary = line > dilated
    vertical_projection = np.sum(binary, axis=0)

    height = line.shape[0]
    whitespace_lengths = []
    whitespace = 0
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            whitespace = whitespace + 1
        elif vp != height:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)
            whitespace = 0
        if index == len(vertical_projection) - 1 and vp == height:
            if whitespace != 0:
                whitespace_lengths.append(whitespace)

    while 1 in whitespace_lengths:
        whitespace_lengths.remove(1)

    avg_white_space_length = np.mean(whitespace_lengths)

    whitespace_length = 0
    divider_indexes = [int(0)]
    for index, vp in enumerate(vertical_projection):
        if vp == height:
            whitespace_length = whitespace_length + 1

        elif vp != height:
            if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                divider_indexes.append(int(index - 2))
            whitespace_length = 0

        if index == len(vertical_projection) - 1:
            divider_indexes.append(int(index))
            whitespace_length = 0

    line_copy = line.copy()
    for i in range(line_copy.shape[0]):
        for j in range(line_copy.shape[1]):
            if j in divider_indexes and j != 0 and j != line_copy.shape[1] - 1:
                line_copy[i][j] = 0

    divider_indexes = np.array(divider_indexes)
    dividers = np.column_stack((divider_indexes[:-1], divider_indexes[1:]))

    words = []
    for index, window in enumerate(dividers):
        words.append(line[:, window[0]:window[1]])

    return words
