"""A data translation layer for handling data read/write to files

File format will be a list of points in the form x,y,z
where z is used to identify any properties of the point
Z Values:
1: Normal Point
2: Joint Point
3: Endpoint
"""
import numpy as np
from character import Character


def write_chars_to_file(char: Character, output_path):
    if not char.usable:
        return

    for i in range(len(char.edges)):
        # only keep every third point in an edge
        char.edges[i] = char.edges[i][::3]

    all_edge_points = np.concatenate(char.edges)
    padded_edge_points = np.zeros((128, 3), np.uint8)
    padded_edge_points[:all_edge_points.shape[0]] = all_edge_points[:128]

    with open(output_path, "w+") as file:
        [file.write("%s,%s,%s\n" % tuple(point)) for point in padded_edge_points]
