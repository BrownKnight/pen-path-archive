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


def write_char_to_file(char: Character, output_path):
    if not char.usable:
        print("Character path is not usable")
        return
    all_edge_points = np.concatenate(char.edges)
    length_of_all_points = len(all_edge_points)
    edges_to_concat = []
    for i in range(len(char.edges)):
        # We want the number of points to be as close to 128 as possible, with a bit of margin
        if length_of_all_points > 250:
            edges_to_concat.append(char.edges[i][::3])
            # Make sure the last point of every edge is still included, as its likely a key point
            if edges_to_concat[-1] != char.edges[i][-1]:
                edges_to_concat[i].append(char.edges[i][-1])
        else:
            edges_to_concat.append(char.edges[i][::2])
            if edges_to_concat[-1] != char.edges[i][-1]:
                edges_to_concat[i].append(char.edges[i][-1])

    all_edge_points = np.concatenate(edges_to_concat)
    padded_edge_points = np.zeros((128, 3), np.uint8)
    padded_edge_points[:all_edge_points.shape[0]] = all_edge_points[:128]

    with open(output_path, "w+") as file:
        [file.write("%s,%s,%s\n" % tuple(point)) for point in padded_edge_points]
