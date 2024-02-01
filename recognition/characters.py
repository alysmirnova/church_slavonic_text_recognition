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
    first_nonzero_row = np.where((thresh_letter == 255).any(axis=1))[0][0]
    return thresh_letter.shape[0] - first_nonzero_row


def cut_double_letters(array_lines):
    total_height = sum(find_height_of_letter(letter) for line in array_lines for word in line for letter in word)
    total_length = sum(letter.shape[1] for line in array_lines for word in line for letter in word)
    total_count = sum(len(word) for line in array_lines for word in line)

    avg_len = total_length / total_count
    avg_height = total_height / total_count

    new_array = []
    for line in array_lines:
        array_words = []
        for word in line:
            array_letters = []
            for letter in word:
                height = find_height_of_letter(letter)
                width = letter.shape[1]

                if width > avg_len * 2 and height < avg_height:
                    amt = round(width / avg_len)
                    split_width = width / amt
                    array_letters.append([letter[:, int(s*split_width):int((s+1)*split_width)] for s in range(amt)])
                else:
                    array_letters.append(letter)

            array_words.append(array_letters)
        new_array.append(array_words)

    return new_array


def resize_image(letter, size=(28, 28)):
    h, w = letter.shape
    if h != w:
        padding = abs(h-w) // 2
        if w > h:
            letter = np.pad(letter, ((padding, padding + (h-w)%2), (0, 0)), mode='constant')
        else:
            letter = np.pad(letter, ((0, 0), (padding, padding + (w-h)%2)), mode='constant')
    return cv2.resize(letter.astype(float), size)


def remove_small_contours(sorted_contours):
    new_contours = [ctr for ctr in sorted_contours if cv2.boundingRect(ctr)[2] * cv2.boundingRect(ctr)[3] > 3]
    return new_contours


 def cropp_letter(letter, a, b, c, d):
    a_index = a - 1 if a != 0 else a
    c_index = c - 1 if c != 0 else c
    return letter[a_index:b+1, c_index:d+1]


def find_parts_of_letters(sorted_contours, letter):
    parts_of_letters = []
    x1, y1, w1, h1 = cv2.boundingRect(sorted_contours[0])
    if len(sorted_contours) == 1:
        cut_letter = letter[y1-1:, :] if y1 != 0 else letter
        parts_of_letters.append(cut_letter)
    else:
        x2, y2, w2, h2 = cv2.boundingRect(sorted_contours[1])
        if len(sorted_contours) == 2:
            if y2 >= y1+h1 and y1+h1 < letter.shape[0]//2:
                if (x1 <= letter.shape[1] // 2 - 2 <= x1+w1) or (x1 <= letter.shape[1] // 2 + 2 <= x1+w1):
                    cut_letter1 = letter[y2-1:, :]
                    cut_letter2 = cropp_letter(letter, y1, y1+h1, x1, x1+w1)
                    parts_of_letters.extend([cut_letter1, cut_letter2])
                else:
                    cut_letter = letter[y2-1:, :]
                    parts_of_letters.append(cut_letter)
            else:
                y = min_sorted_contours(sorted_contours)
                cut_letter = letter[y-1:, :]
                parts_of_letters.append(cut_letter)
        else:
            x3, y3, w3, h3 = cv2.boundingRect(sorted_contours[2])
            if y3 >= y1+h1 and y1+h1 < letter.shape[0]//2 and y3 > y2+h2 and y2+h2 < letter.shape[0]//2:
                mid_line = letter.shape[1] // 2
                if ((x1 <= mid_line - 2 <= x1+w1) or (x1 <= mid_line + 2 <= x1+w1)) and \
                not ((x2 <= mid_line - 2 <= x2+w2) or (x2 <= mid_line + 2 <= x2+w2)):
                    cut_letter1 = letter[y3-1:, :]
                    cut_letter2 = cropp_letter(letter, y1, y1+h1, x1, x1+w1)
                    parts_of_letters.extend([cut_letter1, cut_letter2])
                elif ((x2 <= mid_line - 2 <= x2+w2) or (x2 <= mid_line + 2 <= x2+w2)) and \
                not ((x1 <= mid_line - 2 <= x1+w1) or (x1 <= mid_line + 2 <= x1+w1)):
                    cut_letter1 = letter[y3-1:, :]
                    cut_letter2 = cropp_letter(letter, y2, y2+h2, x2, x2+w2)
                    parts_of_letters.extend([cut_letter1, cut_letter2])
                else:
                    cut_letter1 = letter[y3-1:, :]
                    cut_letter2 = cropp_letter(letter, min(y1, y2), max(y1+h1, y2+h2), min(x1, x2), max(x1+w1, x2+w2))
                    parts_of_letters.extend([cut_letter1, cut_letter2])
            else:
                y = min_sorted_contours(sorted_contours[1:])
                if y >= y1+h1 and y1+h1 < letter.shape[0]//2:
                    cut_letter1 = letter[y-1:, :]
                    cut_letter2 = cropp_letter(letter, y1, y1+h1, x1, x1+w1)
                    parts_of_letters.extend([cut_letter1, cut_letter2])
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

    kernel = np.ones((1, 5), np.uint8)
    
    for i in range(len(new_array_lines)):
        for j in range(len(new_array_lines[i])):
            for k in range(len(new_array_lines[i][j])):
                
                letter = new_array_lines[i][j][k]
                height = find_height_of_letter(letter)
                
                if letter.shape[0] > height:
                    letter = letter[-height:, :]
                    
                dilated = cv2.dilate(letter, kernel, iterations = 1)    
                sorted_contours = find_contours(dilated, 1)
                sorted_contours = remove_small_contours(sorted_contours)
                
                if len(sorted_contours):
                    parts_of_letters = find_parts_of_letters(sorted_contours, letter)
                    parts_of_letters[0] = resize_image(parts_of_letters[0])
                    cut_letter_array = tf.keras.utils.img_to_array(parts_of_letters[0]) / 255.0
                    y_pred = model_loaded.predict(cut_letter_array)
                    y_pred_bool = (y_pred > 0.5)
                    
                    if not (np.all(y_pred_bool == False)):
                        current_line.append(str(CHURCH_SLAVONIC_LETTERS[np.argmax(y_pred)]))
                    
                    if len(parts_of_letters) >= 2:
                        
                        parts_of_letters[1] = resize_image(parts_of_letters[1], (16, 16))
                        cut_sign_array = tf.keras.utils.img_to_array(parts_of_letters[1]) / 255.0
                        y_pred = model_loaded_diac.predict(cut_sign_array, verbose=0)
                        y_pred_bool = (y_pred > 0.5)
                        
                        if not (np.all(y_pred_bool == False)):
                            
                            argmax_y_pred = np.argmax(y_pred)
                            diacritic = str(CHURCH_SLAVONIC_DIACRITICS[argmax_y_pred])
                            last_char = text[-1]

                            special_cases = {
                                5: ("\u0461", diacritic),
                                8: (LETTERS_WITH_TITLE, diacritic),
                                3: ("\u0461", diacritic),
                                6: ("\u0131", diacritic),
                                9: ("\u002E", ':'),
                                9: ("\u002C", ';')
                            }
                            
                            if argmax_y_pred in special_cases:
                                if last_char in special_cases[argmax_y_pred][0] or last_char == special_cases[argmax_y_pred][0]:
                                    text.pop()
                                    text.append(special_cases[argmax_y_pred][1])
                                    continue
                            
                            if last_char in CHURCH_SLAVONIC_VOWELS and argmax_y_pred not in [5, 6, 8, 9]:
                                text.append(diacritic)

            current_line.append(" ")

        current_line = remove_errors_from_text(current_line)
        for symbol in current_line:
            text = text + symbol
        text = text + "\n"
        current_line = []
        text_in_app.SetLabel(text)
