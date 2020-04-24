"""
As the output of the neural network doesn't always keep the correct shape of the letter, this script will look at two
time sequences A (image output) and B (network output), and merge them to create C, when C has the coordinates from A but in the timeline of B.

This is done by looking at each point in B, and moving it to the closest point to it in A
"""
import numpy as np
from math import sqrt


def adopt_path_shape(character_path: np.ndarray, timed_sequence: np.ndarray):
    new_sequence = []
    for index, character_point in enumerate(character_path):
        cx, cy, _ = character_point
        if cx < 1 and cy < 1:
            continue
        new_timestep =  find_closest_point(character_point, timed_sequence)
        new_sequence.append(np.asarray((cx, cy, new_timestep)))
    new_sequence = np.asarray(new_sequence)
    new_sequence = new_sequence[new_sequence[:,2].argsort()]
    new_sequence = new_sequence[:, :2]

    return new_sequence

def find_closest_point(character_path, timed_sequence):
    px, py, _ = character_path
    closest_timestep = 999
    closest_distance = 99999

    for timestep, (x, y) in enumerate(timed_sequence):
        if x < 1 and y < 1:
            continue
        distance = sqrt(abs(px - x) + abs(py - y))
        if distance < closest_distance:
            closest_distance = distance
            closest_timestep = timestep

    return closest_timestep