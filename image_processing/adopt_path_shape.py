"""
As the output of the neural network doesn't always keep the correct shape of the letter, this script will look at two
time sequences A (image output) and B (network output), and merge them to create C, when C has the coordinates from A but in the timeline of B.

This is done by looking at each point in B, and moving it to the closest point to it in A
"""
import numpy as np
from math import sqrt


def adopt_path_shape(character_path: np.ndarray, timed_sequence: np.ndarray):
    timed_sequence = timed_sequence.copy()
    for index, timestep in enumerate(timed_sequence):
        tx, ty = timestep
        if tx < 1 and ty < 1:
            continue
        timed_sequence[index] = find_closest_point(timestep, character_path)

    return timed_sequence

def find_closest_point(timestep, character_path):
    px, py = timestep
    closest_timestep = (0, 0)
    closest_distance = 99999

    for (x, y, _) in character_path:
        if x < 1 and y < 1:
            continue
        distance = sqrt(abs(px - x) + abs(py - y))
        if distance < closest_distance:
            closest_distance = distance
            closest_timestep = (x, y)

    return closest_timestep