import glob
import sys

import cv2
from imutils import resize
from pathlib import Path

from character import Character
from data_handler import write_char_to_file
from globals import SHOW_STEPS
from edges import extract_edges
from skeleton import get_skeleton


def main(input_path, output_path):
    """
    The Image Processing Pipeline

    The pipeline consists of the following steps
    - Process image (resize, blur etc)
    - Skeletonize image to create single-pixel-width edges
    - Extract edge paths for each edge of each letter, where the resultant edges have no intersections
    - Output all the data
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
    success = get_skeleton(character)
    if not success:
        print("Could not get skeleton for %s" % input_path)
        return False

    character.edges = extract_edges(character)

    if SHOW_STEPS:
        cv2.waitKey(1000000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    write_char_to_file(character, output_path)
    return True


def run(mode):
    if mode == 'single':
        print("Operating in single file mode")
        input_path = "test/d.tiff"
        output_path = "test/d.csv"
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


if __name__ == "__main__":
    args = sys.argv
    run_mode = ""
    if len(args) == 2:
        run_mode = args[1]

    run(run_mode)
