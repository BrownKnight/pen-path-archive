from math import sqrt

import cv2
import numpy as np
from imutils.contours import sort_contours
from skimage.morphology import skeletonize

from character import Character
from globals import SHOW_STEPS


def get_skeleton(char: Character):
    # Skeletonize the shapes
    # Skimage function takes image with either True, False or 0,1
    # and returns and image with values 0, 1.

    char.image = cv2.bitwise_not(char.image)
    char.image = char.image == 255
    char.image = skeletonize(char.image)
    char.image = char.image.astype(np.uint8) * 255

    visualisation_image = char.image.copy()

    # Find contours of the skeletons
    contours, hierarchy = cv2.findContours(char.image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sort the contours left-to-right
    contours, _ = sort_contours(contours, "left-to-right")
    if SHOW_STEPS:
        print("Found %s contours" % len(contours))

    if len(contours) == 0:
        return False

    for contour in contours:
        if cv2.arcLength(contour, True) > 40:
            # Take the coordinates of the skeleton points
            rows, cols = np.where(char.image == 255)
            # Add the coordinates to the list
            char.skeleton = list(zip(cols, rows))

            # Find the endpoints for the shape and update a list
            char.endpoints = skeleton_endpoints(char.image)

            # Find the jointpoints for the shape and update a list
            char.jointpoints = skeleton_jointpoints(char.image)

            # Remove any endpoints that are next to jointpoints
            remove_close_points(char.endpoints, char.jointpoints)

            # Draw the endpoints
            [cv2.circle(visualisation_image, ep, 3, 180, 1) for ep in char.endpoints]
            [cv2.circle(visualisation_image, jp, 3, 120, 1) for jp in char.jointpoints]
            if SHOW_STEPS:
                print("Endpoints %s" % char.endpoints)
                print("Jointpoints %s" % char.jointpoints)

            char.add_to_progress_image(char.image, "skeletonized")
            char.add_to_progress_image(visualisation_image, "key points")

    return True


def skeleton_jointpoints(skel):
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[3, 3, 3],
                       [3, 10, 3],
                       [3, 3, 3]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)
    # FOR DEBUGGING
    # cv2.namedWindow('filtered', cv2.WINDOW_NORMAL)
    # cv2.imshow('filtered', filtered)
    # cv2.waitKey(0)

    rows, cols = np.where(filtered == 19)
    coords = list(zip(cols, rows))

    # Remove all jointpoints that are next to each other (i.e. distance < sqrt(2) which we say is ~ 1.5
    remove_close_points(coords, coords)

    return coords


def remove_close_points(coords_to_filter, coords_for_comparison):
    coords_for_comparison = coords_for_comparison.copy()
    for point1 in coords_for_comparison:
        for point2 in coords_to_filter:
            if point2 not in coords_to_filter or point1 == point2:
                continue
            x1, y1 = point1
            x2, y2 = point2
            # Pythagoras to figure out distance
            distance = sqrt((abs(x1 - x2)) ** 2 + (abs(y1 - y2)) ** 2)
            if distance <= 1.5:
                if SHOW_STEPS:
                    print("Joint/End Points %s and %s are very close (%1.2f), removing %s" % (
                        point1, point2, distance, point2))
                coords_to_filter.remove(point2)


def skeleton_endpoints(skel):
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1
    rows, cols = np.where(filtered == 11)
    coords = list(zip(cols, rows))
    return coords
