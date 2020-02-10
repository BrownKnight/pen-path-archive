import sys
from pathlib import Path
from xml.etree import ElementTree
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

from character_bounding_boxes import get_char_bounding_boxes


def convert_points_to_image(file_path):
    """Reads the data from the xml file, and outputs an image encoded in our style showing the pen path"""
    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    test_image = np.zeros((80, 700), np.uint8)
    test_verify_image = np.zeros((200, 1500), np.uint8)
    test_verify_array = []
    plt.ion()
    char_images = [np.zeros((64, 64), np.uint8)]
    char_points = [np.zeros((64,3), dtype='i')]
    x_offset, y_offset, time_offset = 0, 0, 0
    i = 0
    for stroke_set in root.findall('StrokeSet'):
        for stroke in stroke_set.findall('Stroke'):
            x_points, y_points, _ = get_points(stroke)
            points = list(zip(x_points, y_points))
            for point in points:
                test_verify_image[point[1], point[0]] = 255
            print(len(points))
            cv2.imshow('img', test_verify_image)
            cv2.waitKey(1)
            result = input("Add this stroke to previous character?")
            if result == "y":
                add_points_to_array(char_images[i], stroke, x_offset, y_offset, time_offset)
            else:
                char_images.append(np.zeros((64,64), float))
                i = i+1
                x_offset, y_offset, time_offset = add_points_to_array(char_images[i], stroke)

            for point in points:
                test_verify_image[point[1], point[0]] = 127

    for index, image in enumerate(char_images):
        cv2.imwrite("image%s.tif" % index, image)


def add_points_to_array(array, stroke, x_offset=0, y_offset=0, time_offset=0):
    x_points, y_points, time_points = get_points(stroke)
    if x_offset == 0:
        x_offset = min(x_points)
    if y_offset == 0:
        y_offset = min(y_points)
    if time_offset == 0:
        time_offset = min(time_points)

    points = list(zip(x_points - x_offset, y_points - y_offset, time_points - time_offset))

    for point in points:
        array.append((point[0], point[1], point[2]))

    return x_offset, y_offset, time_offset


def get_points(stroke):
    point_tags = stroke.findall('Point')
    points = [(int(int(point.get('x'))/10), int(int(point.get('y'))/10), int(float(point.get('time')) * 100)) for point in point_tags]
    x_points, y_points, time_points = list(zip(*points))
    return np.asarray(x_points), np.asarray(y_points), np.asarray(time_points)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit("Not enough arguments given")

    data_path = str(Path("test/images") / sys.argv[1])

    convert_points_to_image(data_path)
