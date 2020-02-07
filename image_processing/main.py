import sys

import numpy as np
import cv2
from imutils import resize
from pathlib import Path

from data import write_chars_to_file
from globals import SHOW_STEPS, WAIT_TIME
from character_recognition import get_characters
from edges import extract_edges
from skeleton import get_skeletons


def main():
    """The main function of the program, runs the entire pipeline

    The pipeline consists of the following steps
    - Process image (resize, blur etc)
    - Use pytesseract to extract the location of each character, and the character it is
    - Skeletonize image to create single-width edges
    - Extract edge paths for each non-intersecting edge of each letter
    - TODO Something probably needs to happen here (i.e. more key point extraction)
    - Output all the data
    - Inject the data into a DNN
    -
    """
    if len(sys.argv) < 2:
        exit("Not enough arguments given")

    path = Path("test/images") / sys.argv[1]
    print("Reading image from %s" % path)
    img = cv2.imread(str(path), 0)

    # Some smoothing to get rid of the noise
    # img = cv2.bilateralFilter(img, 5, 35, 10)
    img = cv2.GaussianBlur(img, (3, 3), 3)
    # img = cv2.blur(img, (3, 3))

    img = resize(img, width=700)

    process_image = img
    if SHOW_STEPS:
        cv2.imshow("process", process_image)
        cv2.waitKey(WAIT_TIME)

    # Preprocessing to get the shapes
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 11)

    # This needs to happen before the image has its colours inverted to improve the recognition
    th = cv2.bitwise_not(th)
    characters = get_characters(th)

    for char in characters:
        # Invert to highlight the shape
        # char.image = cv2.bitwise_not(char.image)
        # kernel = np.array([[0, 1, 1],
        #                    [0, 1, 0],
        #                    [1, 1, 0]], dtype='uint8')
        # char.image = cv2.morphologyEx(char.image, cv2.MORPH_CLOSE, kernel)
        letter_process_image = char.image
        if SHOW_STEPS:
            cv2.imshow("letter %s " % char.letter, letter_process_image)
            cv2.waitKey(int(WAIT_TIME / 3))

        # Process the image to get the endpoints and skeletons for each letter
        print("Getting skeleton for character %s" % char.letter)
        success = get_skeletons(char)
        if not success:
            print("Skipping the character %s, could not get skeleton" % char.letter)
            continue

        if SHOW_STEPS:
            letter_process_image = np.hstack((letter_process_image, char.image))
            cv2.imshow("letter %s " % char.letter, letter_process_image)
            cv2.waitKey(WAIT_TIME)

        # For each letter, find the edges
        # for each letter etc
        edges_image = np.zeros_like(char.image)
        char.edges = extract_edges(char, edges_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # TODO Write out each character to output folders, for use in the neural network
    write_chars_to_file(characters)

    exit()


def merge_short_edges(e):
    """TODO: NOT USED AT THIS MOMENT, PROBABLY SHOULDN'T BE EITHER"""
    edges = e.copy()
    for i in range(len(edges)):
        for j in range(len(edges)):
            if len(edges[i]) < 5 or len(edges[j]) < 5:
                print("trying to merge a short edge")


if __name__ == "__main__":
    main()
