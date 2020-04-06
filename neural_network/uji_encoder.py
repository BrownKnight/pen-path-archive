import glob
import math
from pathlib import Path

import numpy as np
from cv2 import cv2

IMAGE_INPUT_DIR = "test.nosync/image_input"
IMAGE_OUTPUT_DIR = "test.nosync/image_output"
GROUND_TRUTH_DIR = "test.nosync/ground_truth"


def format_data_file(data_path, data_output_path, char_shrink, offset, rotation):
    chars = []

    with open("%s" % data_path) as file:
        lines = file.readlines()
        # Remove all the comment lines, and empty lines
        lines = [line for line in lines if
                 line != "" and
                 line != "\n" and
                 "COMMENT" not in line and
                 "LEXICON" not in line and
                 "HIERARCHY" not in line and
                 "PEN_UP" not in line and
                 "PEN_DOWN" not in line and
                 "DT 100" not in line
                 ]

        char = []
        for line in lines:
            # Lines are written in the format (x,y,point_type,time_step)
            # Where point_type is 1=normal, 2=stroke_startpoint, 3=stroke_endpoint
            if "SEGMENT" in line:
                if char:
                    chars.append(char)
                char = []
            else:
                x, y = line.strip("\n", ).strip(" ").replace("   ", " ").replace("  ", " ").split(" ")
                # point_type = 2 if mark_as_pen_down else 1
                # mark_as_pen_down = False
                # Divide each coordinate by a given constant to reduce the size of the character
                char.append((int(int(x) / char_shrink), int(int(y) / char_shrink)))

    for index, char in enumerate(chars):
        # Remove any padding from the top-left of the points to reduce the image size
        # Add 2 to the offset to give each character a suitable border for the image processing
        x_offset = min([point[0] for point in char]) - offset
        y_offset = min([point[1] for point in char]) - offset

        # Apply an offset to each point
        char = [(x - x_offset, y - y_offset) for x, y in char]

        # Add a rotation to each point
        char = [rotate_coords(x, y, rotation) for x, y in char]

        if any([x > 62 or y > 62 or x < 0 or y < 0 for x, y in char]):
            # print('Image too big!! %s skipping' % index)
            pass
        else:
            padded_points = np.zeros((128, 2), int)
            padded_points[:len(char)] = char[:128]

            with open(data_output_path % index, "w+") as file:
                #  file.writelines(["%s,%s,%s\n" % tuple(point) for point in padded_points])
                file.writelines(["%s,%s\n" % tuple(point) for point in padded_points])



def rotate_coords(x, y, rotation):
    angle = math.radians(rotation)

    ox, oy = 32, 32
    px, py = x, y

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy


def create_image_from_file(image_path, image_output_path):
    with open(image_path) as file:
        lines = file.readlines()
        lines = [line.strip("\n") for line in lines]
        points = [(int(point.split(",")[0]), int(point.split(",")[1])) for point in lines]

    char = np.zeros((64, 64), np.uint8)
    for i in range(len(points) - 1):
        if points[i] != (-1, -1) and points[i + 1] != (-1, -1) and points[i] != (0, 0) and points[i + 1] != (0, 0):
            cv2.line(char, points[i], points[i + 1], 255, 2)

    char = cv2.bitwise_not(char)
    cv2.imwrite(image_output_path, char)

    return char


def main():
    # Create multiple iterations of all 11 data files with various levels of char_shrink and offset
    for i in range(1, 12, 1):
        print("Creating ground truth files for #%s" % i)
        # Generate all the different ground truth files for this data file
        for shrink in range(10, 27, 2):
            for offset in range(4, 17, 2):
                for rotation in [270, 340, 20, 90]:
                    format_data_file("original_data/UJIpenchars-w%02d" % i,
                                     "%s/char-%02d" % (GROUND_TRUTH_DIR, i) + "-%03d-" + "s%02d-o%02d-r%03d.txt" % (
                                         shrink, offset, rotation),
                                     char_shrink=shrink,
                                     offset=offset,
                                     rotation=rotation)

        # Create the image files for these generated ground truth files
        print("Creating image files for #%s" % i)
        for file_path in glob.glob("%s/char-%02d-*.txt" % (GROUND_TRUTH_DIR, i)):
            output_path = "%s/%s.tif" % (IMAGE_INPUT_DIR, Path(file_path).stem)
            create_image_from_file(file_path, output_path)


if __name__ == "__main__":
    Path(GROUND_TRUTH_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    main()

    # for i in [0, 15, 30, 330, 345]:
    #     data_output_path = "test/ground_truth/char-%03d-" + "%s.txt" % i
    #     format_data_file("original_data/UJIpenchars-w01",
    #                      data_output_path,
    #                      16, 10, i
    #                      )
    # for file_path in sorted(glob.glob("test/ground_truth/char-000-*.txt")):
    #     output_path = "test/image_input/%s.tif" % Path(file_path).stem
    #     create_image_from_file(file_path, output_path)
