import sys
import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from recognition.alphabet import CHURCH_SLAVONIC_LETTERS, CHURCH_SLAVONIC_DIACRITICS, CHURCH_SLAVONIC_VOWELS,\
    LETTERS_WITH_TITLE
from segmentation.thresholding import thresholding
from segmentation.letters import min_sorted_contours, find_contours
from keras import backend as K
K.set_image_data_format('channels_first')


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


def find_height_of_letter(letter):
    thresh_letter = thresholding(letter, False)
    i = 0
    f = True
    while f and i < thresh_letter.shape[0]:
        if np.count_nonzero(thresh_letter[i] == 255) != 0:
            f = False
        else:
            i += 1
    return thresh_letter.shape[0] - i


def cut_double_letters(array_lines):
    suma_len = 0
    suma_height = 0
    count = 0
    for i in range(len(array_lines)):
        for j in range(len(array_lines[i])):
            for k in range(len(array_lines[i][j])):
                letter = array_lines[i][j][k]
                suma_height += find_height_of_letter(letter)
                suma_len += letter.shape[1]
                count += 1
    avg_len = suma_len / count
    avg_height = suma_height / count
    new_array = []
    for i in range(len(array_lines)):
        array_words = []
        for j in range(len(array_lines[i])):
            array_letters = []
            for k in range(len(array_lines[i][j])):
                letter = array_lines[i][j][k]
                height = find_height_of_letter(letter)
                width = letter.shape[1]
                if width > avg_len * 2 and height < avg_height:
                    amt = round(width / avg_len)
                    for s in range(0, amt):
                        if int(s * width / amt) != 0:
                            array_letters.append(letter[:, int(s * width / amt) - 1:int((s + 1)
                                                                                        * width / amt) + 1])
                        else:
                            array_letters.append(letter[:, int(s * width / amt):int((s + 1)
                                                                                    * width / amt) + 1])
                else:
                    array_letters.append(letter)
            array_words.append(array_letters)
        new_array.append(array_words)
    return new_array


def resize_image(letter, size=(28, 28)):
    h, w = letter.shape
    if h != w:
        if w > h:
            add_top = (w-h) // 2 if (w-h) % 2 == 0 else (w-h) // 2 + 1
            add_bottom = (w-h) // 2
            array = [0 for _ in range(w)]
            ndarray = np.array(array, dtype=int)
            for _ in range(add_top):
                letter = np.vstack((ndarray, letter))
            for _ in range(add_bottom):
                letter = np.vstack((letter, ndarray))
        else:
            add_left = (h-w) // 2 if (h-w) % 2 == 0 else (h-w) // 2 + 1
            add_right = (h-w) // 2
            nd = np.array([0])
            for _ in range(h-1):
                nd = np.vstack((nd, [0]))
            for _ in range(add_left):
                letter = np.insert(letter, 0, [0], axis=1)
            for _ in range(add_right):
                letter = np.append(letter, nd, axis=1)
    return cv2.resize(letter.astype(float), size)


def remove_small_contours(sorted_contours):
    new_contours = []
    for i in range(len(sorted_contours)):
        x, y, w, h = cv2.boundingRect(sorted_contours[i])
        if ((x+w)-x) * ((y+h)-y) > 3:
            new_contours.append(sorted_contours[i])
    return new_contours


def cropp_letter(letter, a, b, c, d):
    if c != 0:
        return letter[a-1:b+1, c-1:d+1] if a != 0 else letter[a:b+1, c-1:d+1]
    else:
        return letter[a-1:b+1, c:d+1] if a != 0 else letter[a:b+1, c:d+1]


def find_parts_of_letters(sorted_contours, letter):
    parts_of_letters = []
    if len(sorted_contours) == 1:
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        cut_letter = letter[y-1:, :] if y != 0 else letter
        parts_of_letters.append(cut_letter)
    else:
        if len(sorted_contours) == 2:
            x1, y1, w1, h1 = cv2.boundingRect(sorted_contours[0])
            x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[1])
            if y2 >= y1+h1 and y1+h1 < letter.shape[0]//2:
                if (x1 <= letter.shape[1] // 2 - 2 <= x1+w1) or (x1 <= letter.shape[1] // 2 + 2 <= x1+w1):
                    cut_letter1 = letter[y2-1:, :]
                    parts_of_letters.append(cut_letter1)
                    cut_letter2 = cropp_letter(letter, y1, y1+h1, x1, x1+w1)
                    parts_of_letters.append(cut_letter2)
                else:
                    cut_letter = letter[y2-1:, :]
                    parts_of_letters.append(cut_letter)
            else:
                y = min_sorted_contours(sorted_contours)
                cut_letter = letter[y-1:, :]
                parts_of_letters.append(cut_letter)
        else:
            x1, y1, w1, h1 = cv2.boundingRect(sorted_contours[0])
            x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[1])
            x3, y3, w3, h3 = cv2.boundingRect(sorted_contours[2])
            if y3 >= y1+h1 and y1+h1 < letter.shape[0]//2 and y3 > y2+h2 and y2+h2 < letter.shape[0]//2:
                if ((x1 <= letter.shape[1] // 2 - 2 <= x1+w1) or (x1 <= letter.shape[1] // 2 + 2 <= x1+w1))\
                        and not ((x2 <= letter.shape[1] // 2 - 2 <= x2+w2)
                                 or (x2 <= letter.shape[1] // 2 + 2 <= x2+w2)):
                    cut_letter1 = letter[y3-1:, :]
                    parts_of_letters.append(cut_letter1)
                    cut_letter2 = cropp_letter(letter, y1, y1+h1, x1, x1+w1)
                    parts_of_letters.append(cut_letter2)
                elif ((x2 <= letter.shape[1] // 2 - 2 <= x2+w2) or (x2 <= letter.shape[1] // 2 + 2 <= x2+w2))\
                        and not ((x1 <= letter.shape[1] // 2 - 2 <= x1+w1)
                                 or (x1 <= letter.shape[1] // 2 + 2 <= x1+w1)):
                    cut_letter1 = letter[y3-1:, :]
                    parts_of_letters.append(cut_letter1)
                    cut_letter2 = cropp_letter(letter, y2, y2+h2, x2, x2+w2)
                    parts_of_letters.append(cut_letter2)
                else:
                    cut_letter1 = letter[y3-1:, :]
                    parts_of_letters.append(cut_letter1)
                    cut_letter2 = cropp_letter(letter, min(y1, y2), max(y1+h1, y2+h2), min(x1, x2),
                                               max(x1+w1, x2+w2))
                    parts_of_letters.append(cut_letter2)
            else:
                y = min_sorted_contours(sorted_contours[1:])
                if y >= y1+h1 and y1+h1 < letter.shape[0]//2:
                    cut_letter1 = letter[y-1:, :]
                    parts_of_letters.append(cut_letter1)
                    cut_letter2 = cropp_letter(letter, y1, y1+h1, x1, x1+w1)
                    parts_of_letters.append(cut_letter2)
                else:
                    y = min_sorted_contours(sorted_contours)
                    cut_letter = letter[y-1:, :]
                    parts_of_letters.append(cut_letter)
    return parts_of_letters


def connect_cut_letters(text_with_corrections, letter, next_letter=True):
    text_with_corrections.pop()
    if next_letter:
        text_with_corrections.pop()
    text_with_corrections.append(letter)


def remove_errors_from_text(text):
    text_with_corrections = [text[0], text[1]]
    cut_letter_sets = [["\u044C", "\u0131", "\u044B"], ["\u0131", "\u0131", "\u043D"],
                       ["\u002C", "\u0131", "\u0447"], ["\u0446", "\u0131", "\u0449"]]

    for i in range(2, len(text)):

        if text[i] not in CHURCH_SLAVONIC_DIACRITICS:
            for cls in cut_letter_sets:
                if text[i - 2] == cls[0] and text[i - 1] == cls[1]:
                    connect_cut_letters(text_with_corrections, cls[2])

        flag = False
        if i == len(text) - 1:
            for cls in cut_letter_sets:
                if text[i - 1] == cls[0] and text[i] == cls[1]:
                    connect_cut_letters(text_with_corrections, cls[2], False)
                    flag = True
            if flag:
                continue

        if text[i - 1] == "\u0131" and text[i] == "\u0440":
            text_with_corrections.pop()
            text_with_corrections.append("\u0449")
            continue

        if text[i - 1] == " " and text[i] in [",", ":", "\n"]:
            text_with_corrections.pop()
        text_with_corrections.append(text[i])
    return text_with_corrections


def character_recognition(array_lines, text_in_app):
    model_loaded = keras.models.load_model(resource_path('model_letters.h5'))
    model_loaded_diac = keras.models.load_model(resource_path('model_diac.h5'))

    text = ''
    current_line = []
    new_array_lines = cut_double_letters(array_lines)
    for i in range(len(new_array_lines)):
        for j in range(len(new_array_lines[i])):
            for k in range(len(new_array_lines[i][j])):
                kernel = np.ones((1, 5), np.uint8)
                letter = new_array_lines[i][j][k]
                height = find_height_of_letter(letter)
                if letter.shape[0] - height != 0:
                    letter = letter[letter.shape[0] - height - 1:, :]
                dilated = cv2.dilate(letter, kernel, iterations=1)
                sorted_contours = find_contours(dilated, 1)
                sorted_contours = remove_small_contours(sorted_contours)
                if len(sorted_contours) != 0:
                    parts_of_letters = find_parts_of_letters(sorted_contours, letter)
                    parts_of_letters[0] = resize_image(parts_of_letters[0])
                    cut_letter_array = tf.keras.utils.img_to_array(parts_of_letters[0])
                    cut_letter_array /= 255
                    y_pred = model_loaded.predict(cut_letter_array, verbose=0)
                    y_pred_bool = (y_pred > 0.5)
                    if not (np.all(y_pred_bool == False)):
                        current_line.append(str(CHURCH_SLAVONIC_LETTERS[np.argmax(y_pred)]))
                    excluded = [5, 6, 8, 9]
                    if len(parts_of_letters) >= 2:
                        parts_of_letters[1] = resize_image(parts_of_letters[1], (16, 16))
                        cut_sign_array = tf.keras.utils.img_to_array(parts_of_letters[1])
                        cut_sign_array /= 255
                        y_pred = model_loaded_diac.predict(cut_sign_array, verbose=0)
                        y_pred_bool = (y_pred > 0.5)
                        if not (np.all(y_pred_bool == False)):
                            if np.argmax(y_pred) == 5 and current_line[-1] == "\u0461":
                                current_line.pop()
                                current_line.append(str(CHURCH_SLAVONIC_DIACRITICS[np.argmax(y_pred)]))
                                continue
                            if np.argmax(y_pred) == 8 and current_line[-1] in LETTERS_WITH_TITLE:
                                current_line.append(str(CHURCH_SLAVONIC_DIACRITICS[np.argmax(y_pred)]))
                                continue
                            if np.argmax(y_pred) == 3 and current_line[-1] == "\u0461":
                                current_line.append(str(CHURCH_SLAVONIC_DIACRITICS[np.argmax(y_pred)]))
                                continue
                            if np.argmax(y_pred) == 6 and current_line[-1] == "\u0131":
                                current_line.append(str(CHURCH_SLAVONIC_DIACRITICS[np.argmax(y_pred)]))
                                continue
                            if np.argmax(y_pred) == 9 and current_line[-1] == "\u002E":
                                current_line.pop()
                                current_line.append(':')
                                continue
                            if np.argmax(y_pred) == 9 and current_line[-1] == "\u002C":
                                current_line.pop()
                                current_line.append(';')
                                continue
                            if current_line[-1] in CHURCH_SLAVONIC_VOWELS and np.argmax(y_pred) not in excluded:
                                current_line.append(str(CHURCH_SLAVONIC_DIACRITICS[np.argmax(y_pred)]))

            current_line.append(" ")

        current_line = remove_errors_from_text(current_line)
        for symbol in current_line:
            text = text + symbol
        text = text + "\n"
        current_line = []
        text_in_app.SetLabel(text)
