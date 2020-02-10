"""Using pytesseract, we can extract each individual character and understand what char it is"""

import csv
import cv2
from imutils import resize
import numpy as np
from imutils.contours import sort_contours

from pytesseract import pytesseract as pt, Output
import sys

from skimage.morphology import skeletonize

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

    get_char_bounding_boxes(img)


def get_char_bounding_boxes(img):
    """TODO If we use the skeletonize approach for this function, it should be merged with skeleton.py"""
    chars = []

    # Skeletonize the shapes
    # Skimage function takes image with either True, False or 0,1
    # and returns and image with values 0, 1.

    skeleton_img = cv2.bitwise_not(img)
    skeleton_img = skeleton_img == 255
    skeleton_img = skeletonize(skeleton_img)
    skeleton_img = skeleton_img.astype(np.uint8) * 255

    # Find contours of the skeletons
    contours, hierarchy = cv2.findContours(skeleton_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sort the contours left-to-right
    contours, _ = sort_contours(contours, "left-to-right")
    for index, contour in enumerate(contours):
        if cv2.arcLength(contour, True) > 40:
            # Bounding rect of the contour
            x, y, w, h = cv2.boundingRect(contour)
            char = Character()
            char.image = img[y - 1:y + h + 1, x - 1:x + w + 1]
            char.image = cv2.copyMakeBorder(char.image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, 255)
            char.image = cv2.resize(char.image, (64, 64))
            char.letter = str(index)
            chars.append(char)

    return chars


# def get_characters(img):
#     boxes = pt.image_to_boxes(img, output_type=Output.DICT)
#     if boxes is None or boxes['char'] == ['']:
#         exit("No image_input could be found")
#     print(boxes)
#
#     display_bounding_box_image = img.copy()
#
#     # Extract each individual letter and Draw the bounding box on the image
#     h, w = img.shape
#     chars = []
#     for index in range(len(boxes['char'])):
#         char = Character()
#         char.letter = boxes['char'][index]
#
#         left = int(boxes['left'][index]) - 1
#         top = h - int(boxes['top'][index]) - 1
#         right = int(boxes['right'][index]) + 1
#         bottom = h - int(boxes['bottom'][index]) + 1
#
#         char.image = img[top:bottom, left:right]
#         # Add a plain border around the image to improve skeleton recognition
#         char.image = cv2.copyMakeBorder(char.image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, None, 255)
#         # Make sure the array is copied to ensure changes to it for display
#         # purposes dont affect the actual letter representation
#         char.image = char.image.copy()
#         chars.append(char)
#
#         cv2.rectangle(display_bounding_box_image, (left, top), (right, bottom), 0, 2)
#
#     if SHOW_STEPS:
#         cv2.imshow('image_output', display_bounding_box_image)
#         cv2.waitKey(WAIT_TIME)
#
#     return chars


if __name__ == "__main__":
    test_main()
