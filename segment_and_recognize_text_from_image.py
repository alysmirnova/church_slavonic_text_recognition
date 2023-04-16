from segmentation.lines import line_segmentation
from segmentation.words import word_segmentation
from segmentation.letters import letter_segmentation
from recognition.characters import character_recognition


def segment_and_recognize(img):
    line_images = line_segmentation(img)
    array_lines = []
    for i in range(len(line_images)):
        array_words = []
        words = word_segmentation(line_images[i])
        for j in range(len(words)):
            array_letters = []
            letters = letter_segmentation(words[j])
            for k in range(len(letters)):
                array_letters.append(letters[k])
            array_words.append(letters)
        array_lines.append(array_words)

    text_from_image = character_recognition(array_lines)
    text = ''
    for symbol in text_from_image:
        text = text + symbol
    return text
