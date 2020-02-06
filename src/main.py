import sys

import numpy as np
import cv2
from imutils import resize

from globals import SHOW_STEPS, WAIT_TIME
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

    path = sys.argv[1]
    print("Reading image from %s" % path)
    img = cv2.imread(path, 0)

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

    # Invert to highlight the shape
    th = cv2.bitwise_not(th)
    kernel = np.array([[0, 1, 1],
                       [0, 1, 0],
                       [1, 1, 0]], dtype='uint8')
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    if SHOW_STEPS:
        process_image = np.hstack((process_image, th))
        cv2.imshow('process', process_image)

    # Process the image to get the endpoints and skeletons for each letter
    endpoints, jointpoints, letter_skeletons, mask, th = get_skeletons(th)

    if SHOW_STEPS:
        process_image = np.hstack((process_image, th))
        cv2.imshow('process', process_image)
        cv2.waitKey(0)

    # For each letter, find the edges
    # for each letter etc
    letter = []
    edges_image = np.zeros_like(mask)
    for index in range(len(letter_skeletons)):
        letter.append(extract_edges(endpoints[index], jointpoints[index], letter_skeletons[index], edges_image))

    edges_image[-25:, -25:] = 255
    cv2.imshow('letters', edges_image)
    print("Found edges for %s letters" % len(letter))

    cv2.waitKey(0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()


def merge_short_edges(e):
    edges = e.copy()
    for i in range(len(edges)):
        for j in range(len(edges)):
            if len(edges[i]) < 5 or len(edges[j]) < 5:
                print("trying to merge a short edge")


if __name__ == "__main__":
    main()
