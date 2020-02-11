import glob
from pathlib import Path

import numpy as np
from cv2 import cv2

CHARACTER_SHRINK_MULTIPLIER = 16


def format_data_file(data_path, data_output_path):
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
                 "PEN_DOWN" not in line
                 ]

        char = []
        for line in lines:
            if "SEGMENT" in line:
                if char:
                    chars.append(char)
                char = []

            elif "DT 100" in line:
                # (-1,-1) will mean a PEN_UP action, delimiting between multiple strokes for the same character
                char.append((-1, -1))

            else:
                x, y = line.strip("\n", ).strip(" ").replace("   ", " ").replace("  ", " ").split(" ")
                # Divide each coordinate by a given constant to reduce the size of the character
                char.append((int(int(x) / CHARACTER_SHRINK_MULTIPLIER), int(int(y) / CHARACTER_SHRINK_MULTIPLIER)))

    average_char_length = sum([len(char) for char in chars]) / len(chars)
    print(average_char_length)
    print(max([len(char) for char in chars]))

    for index, char in enumerate(chars):
        # Remove any padding from the top-left of the points to reduce the image size
        # Add 2 to the offset to give each character a suitable border for the image processing
        x_offset = min([point[0] for point in char if point != (-1, -1)]) - 2
        y_offset = min([point[1] for point in char if point != (-1, -1)]) - 2

        char = [(point[0] - x_offset, point[1] - y_offset, i)
                if point != (-1, -1) else (-1, -1, i)
                for i, point in enumerate(char)]

        if any([point[0] > 62 or point[1] > 62 for point in char]):
            print('Image too big!! %s skipping' % index)
        else:
            padded_points = np.zeros((128, 3), int)
            padded_points[:len(char)] = char[:128]

            with open(data_output_path % index, "w+") as file:
                file.writelines(["%s,%s,%s\n" % (point[0], point[1], point[2]) for point in padded_points])


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
    format_data_file("test/original_data/UJIpenchars-w11", "test/ground_truth/character11-%s.txt")

    for file_path in glob.glob("test/ground_truth/*.txt"):
        output_path = "test/image_input/%s.tif" % Path(file_path).stem
        create_image_from_file(file_path, output_path)
