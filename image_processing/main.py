import glob
import sys

import numpy as np
import cv2
from imutils import resize
from pathlib import Path

from character import Character
from data_handler import write_chars_to_file
from globals import SHOW_STEPS, WAIT_TIME
from character_bounding_boxes import get_char_bounding_boxes
from edges import extract_edges
from skeleton import get_skeletons


def main(input_path, output_path):
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
    img = cv2.imread(input_path, 0)

    # Some smoothing to get rid of the noise
    # img = cv2.bilateralFilter(img, 5, 35, 10)
    # img = cv2.GaussianBlur(img, (2, 2), 3)
    # img = cv2.blur(img, (3, 3))

    img = resize(img, width=64)

    character = Character()
    character.progress_image = img
    if SHOW_STEPS:
        cv2.namedWindow("progress", flags=cv2.WINDOW_GUI_EXPANDED | cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Preprocessing to get the shapes
    threshold_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 35, 11)

    character.image = threshold_image

    character.add_to_progress_image(threshold_image, "Threshold")

    # Process the image to get the endpoints and skeletons for each letter
    success = get_skeletons(character)
    if not success:
        exit("Could not get skeleton")

    character.edges = extract_edges(character)

    if SHOW_STEPS:
        cv2.waitKey(1000000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    # TODO Write out each character to image_output folders, for use in the neural network
    write_chars_to_file(character, output_path)


if __name__ == "__main__":
    args = sys.argv
    mode = ""
    if len(args) == 2:
        mode = args[1]

    if mode == 'single':
        print("Operating in single file mode")
        input_path = "test.nosync/image_input/char-01-000-12-04.tif"
        output_path = "test.nosync/image_output/char-01-000-12-04.csv"
        print("Reading image from %s" % input_path)
        print("Outputting image data to %s" % output_path)

        main(input_path, output_path)
    elif mode == 'directory':
        print("Operating in directory mode")
        file_paths = sorted(glob.glob("test.nosync/image_input/*.tif"))
        for index, file in enumerate(file_paths):
            if index % 100 == 0:
                print("Processing file %s" % file)
            output_path = "test.nosync/image_output/%s.csv" % Path(file).stem
            main(file, output_path)
    else:
        exit("Incorrect arguments given. Supports args: 'single' | 'directory")
