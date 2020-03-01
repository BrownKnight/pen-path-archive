import glob
from pathlib import Path

import numpy as np
from cv2 import cv2


def format_data_file(data_path, data_output_path, char_shrink, offset):
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
                 "DT 100" not in line
                 ]

        char = []
        mark_as_pen_down = False
        for line in lines:
            # Lines are written in the format (x,y,point_type,time_step)
            # Where point_type is 1=normal, 2=stroke_startpoint, 3=stroke_endpoint
            if "SEGMENT" in line:
                if char:
                    chars.append(char)
                char = []

            elif "PEN_UP" in line:
                # char[-1] = (char[-1][0], char[-1][1], 3)
                print("Do Nothing")

            elif "PEN_DOWN" in line:
                # mark_as_pen_down = True
                print("Do Nothing")

            else:
                x, y = line.strip("\n", ).strip(" ").replace("   ", " ").replace("  ", " ").split(" ")
                # point_type = 2 if mark_as_pen_down else 1
                # mark_as_pen_down = False
                # Divide each coordinate by a given constant to reduce the size of the character
                char.append((int(int(x) / char_shrink), int(int(y) / char_shrink)))

    average_char_length = sum([len(char) for char in chars]) / len(chars)
    print(average_char_length)
    print(max([len(char) for char in chars]))

    for index, char in enumerate(chars):
        # Remove any padding from the top-left of the points to reduce the image size
        # Add 2 to the offset to give each character a suitable border for the image processing
        x_offset = min([point[0] for point in char]) - offset
        y_offset = min([point[1] for point in char]) - offset

        char = [(point[0] - x_offset, point[1] - y_offset, position) for position, point in enumerate(char)]

        if any([point[0] > 62 or point[1] > 62 for point in char]):
            print('Image too big!! %s skipping' % index)
        else:
            padded_points = np.zeros((128, 3), int)
            padded_points[:len(char)] = char[:128]

            with open(data_output_path % index, "w+") as file:
                file.writelines(["%s,%s,%s\n" % tuple(point) for point in padded_points])


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


if __name__ == "__main__":
    char_shrink_multipliers = [12, 14, 16, 18, 20, 22, 24, 26]
    # Create 5 iterations of all 11 files
    for i in range(1, 12, 1):
        print("Processing file #%s" % i)
        iteration = 0
        for shrink in char_shrink_multipliers:
            for k in range(2, 11, 2):
                format_data_file("test/original_data/UJIpenchars-w%02d" % i,
                                 "test/ground_truth/char-%02d-%02d" % (i, iteration) + "-%03d.txt",
                                 char_shrink=shrink,
                                 offset=k)
                iteration += 1

    # format_data_file("test/original_data/UJIpenchars-w01", "test/ground_truth/character01-1-%03d.txt")

    for file_path in glob.glob("test/ground_truth/*.txt"):
        output_path = "test/image_input/%s.tif" % Path(file_path).stem
        create_image_from_file(file_path, output_path)
