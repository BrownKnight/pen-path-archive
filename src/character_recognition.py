"""Using pytesseract, we can extract each individual character and understand what char it is"""

import csv
import cv2
from imutils import resize
import numpy as np

from pytesseract import pytesseract as pt, Output
import sys

from character import Character
from globals import SHOW_STEPS, WAIT_TIME


def test_main():
    if len(sys.argv) < 2:
        exit("Not enough arguments given")
    path = sys.argv[1]
    print("Reading image from %s" % path)
    img = cv2.imread(path, 0)
    # Some smoothing to get rid of the noise
    # img = cv2.bilateralFilter(img, 5, 35, 10)
    img = cv2.GaussianBlur(img, (3, 3), 3)
    # img = cv2.blur(img, (3, 3))
    img = resize(img, width=700)
    # Preprocessing to get the shapes
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, 11)
    # Invert to highlight the shape
    # img = cv2.bitwise_not(img)
    # kernel = np.array([[0, 1, 1],
    #                    [0, 1, 0],
    #                    [1, 1, 0]], dtype='uint8')
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    get_characters(img)


def get_characters(img):
    boxes = pt.image_to_boxes(img, output_type=Output.DICT)
    if boxes is None or boxes['char'] == ['']:
        exit("No characters could be found")
    print(boxes)

    # Extract each individual letter and Draw the bounding box on the image
    h, w = img.shape
    chars = []
    for index in range(len(boxes['char'])):
        char = Character()
        char.letter = boxes['char'][index]

        left = int(boxes['left'][index])
        top = h - int(boxes['top'][index])
        right = int(boxes['right'][index])
        bottom = h - int(boxes['bottom'][index])

        char.image = img[top:bottom, left:right]
        # Make siure the array is copied to ensure changes to it for display
        # purposes dont affect the actual letter representation
        char.image = char.image.copy()
        chars.append(char)

        cv2.rectangle(img, (left, top), (right, bottom), 0, 2)

    if SHOW_STEPS:
        cv2.imshow('output', img)
        cv2.waitKey(WAIT_TIME)

    return chars


if __name__ == "__main__":
    test_main()
