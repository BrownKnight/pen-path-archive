"""
When using the system, it can often be easier to input an image with multiple characters on to get paths for each
character. TO do this, before processing any characters we should first look at the image create a new image for each
character.

This is done by:
Pre-processing the image (blur, threshold)
Extracting contours from the image
taking the bounding box of each contour
"""
import numpy as np
import cv2
from imutils.contours import sort_contours
from imutils import resize
from math import floor, ceil

from globals import SHOW_STEPS, WAIT_TIME


def get_bounding_boxes(image_path, output_dir):
    multi_char_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if SHOW_STEPS:
        cv2.namedWindow("Process", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_FREERATIO)
        cv2.imshow("Process", multi_char_image)
        cv2.waitKey(WAIT_TIME)

    # Image Pre-Processing:
    # Invert, Blur, Threshold
    cv2.adaptiveThreshold(multi_char_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11,
                          dst=multi_char_image)

    if SHOW_STEPS:
        cv2.imshow("Process", multi_char_image)
        cv2.waitKey(WAIT_TIME)


    cv2.blur(multi_char_image, (3, 3), multi_char_image)
    if SHOW_STEPS:
        cv2.imshow("Process", multi_char_image)
        cv2.waitKey(WAIT_TIME)

    # Make a copy of the image for saving later, without any info or extra processing drawn onto it
    original_image = multi_char_image.copy()

    cv2.bitwise_not(multi_char_image, multi_char_image)
    if SHOW_STEPS:
        cv2.imshow("Process", multi_char_image)
        cv2.waitKey(WAIT_TIME)

    # Get the contours out of the image
    contours, _ = cv2.findContours(multi_char_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, _ = sort_contours(contours, 'left-to-right')

    # Get the bounding boxes for each contour
    boxes = []
    for contour in contours:
        if cv2.arcLength(contour, True) < 40:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
        if SHOW_STEPS:
            cv2.rectangle(multi_char_image, (x, y), (x + w, y + h), 150, 1)

    if SHOW_STEPS:
        cv2.imshow("Process", multi_char_image)
        cv2.waitKey(WAIT_TIME)

    outer_boxes = []
    # Look for overlapping boxes, and if they do overlap then merge them
    for box1 in boxes:
        for box2 in boxes:
            if do_boxes_overlap(box1, box2):
                outer_boxes.append(merge_boxes(box1, box2))
                boxes.remove(box1)
                boxes.remove(box2)

    outer_boxes += boxes
    for box in outer_boxes:
        x, y, x1, y1 = box
        if SHOW_STEPS:
            cv2.rectangle(multi_char_image, (x, y), (x1, y1), 255, 2)

    if SHOW_STEPS:
        cv2.imshow("Process", multi_char_image)
        cv2.waitKey(WAIT_TIME)
        cv2.waitKey(0)

    # Take each bounding box and make it a 64x64 image with at least a 4 pixel border
    chars = []
    for box in outer_boxes:
        x, y, x1, y1 = box
        char = original_image[y:y1, x:x1]

        # Resize the image until the longest side is 52px
        h, _ = char.shape
        if h > 52:
            char = resize(char, height=52)
        _, w = char.shape
        if w > 52:
            char = resize(char, width=52)

        # Fill in the image until it is a 64x64 square
        h, w = char.shape
        h_gap = 64-h
        w_gap = 64-w
        top = floor(h_gap/2)
        bottom = ceil(h_gap/2)
        left = ceil(w_gap/2)
        right = floor(w_gap/2)

        char = cv2.copyMakeBorder(char, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=255)
        print(char.shape)
        chars.append(char)

    for index, char in enumerate(chars):
        cv2.imwrite(output_dir + "/char-%03d.tif" % index, char)


def do_boxes_overlap(box1, box2):
    box1_x, box1_y, box1_x1, box1_y1 = box1
    box2_x, box2_y, box2_x1, box2_y1 = box2

    # Check if top-left in box2
    if box2_x < box1_x < box2_x1 and box2_y < box1_y < box2_y1:
        return True

    # Check if top-right in box2
    if box2_x < box1_x1 < box2_x1 and box2_y < box1_y < box2_y1:
        return True

    # Check if bottom_right in box2
    if box2_x < box1_x1 < box2_x1 and box2_y < box1_y1 < box2_y1:
        return True

    # Check if bottom_left in box2
    if box2_x < box1_x < box2_x1 and box2_y < box1_y1 < box2_y1:
        return True

    return False


def merge_boxes(box1, box2):
    box1_x, box1_y, box1_x1, box1_y1 = box1
    box2_x, box2_y, box2_x1, box2_y1 = box2
    x = min(box1_x, box2_x)
    y = min(box1_y, box2_y)
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    return x, y, x1, y1



if __name__ == '__main__':
    get_bounding_boxes("test/multi_char/all_chars.jpeg", "test/image_input")
