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

from character import Character


def write_chars_to_file(chars: List[Character], output_path):
    for char in chars:
        if not char.usable:
            continue

        all_edge_points = np.concatenate(char.edges)
        padded_edge_points = np.zeros((128, 3), np.uint8)
        padded_edge_points[:all_edge_points.shape[0]] = all_edge_points[:128]

        with open(output_path % char.letter, "w+") as file:
            [file.write("%s,%s,%s\n" % (point[0], point[1], point[2])) for point in padded_edge_points]
