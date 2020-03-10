"""A data translation layer for handling data read/write to files

File format will be a list of points in the form x,y,z
where z is used to identify any properties of the point
Z Values:
64: Normal Point
128: Joint Point
192: Endpoint
"""
import csv
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from character import Character


def write_chars_to_file(char: Character, output_path):
    if not char.usable:
        return

    all_edge_points = np.concatenate(char.edges)
    padded_edge_points = np.zeros((128, 3), np.uint8)
    padded_edge_points[:all_edge_points.shape[0]] = all_edge_points[:128]

    with open(output_path, "w+") as file:
        print("Writing to %s" % output_path)
        # image = np.zeros((64, 64))
        # for point in padded_edge_points:
        #     image[int(point[1]), int(point[0])] = float(point[2])
        # print("displaying image")
        # plt.imshow(image)
        # plt.show()
        [file.write("%s,%s,%s\n" % tuple(point)) for point in padded_edge_points]
