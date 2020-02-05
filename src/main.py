from math import sqrt

import numpy as np
import cv2
from imutils import resize
from imutils.contours import sort_contours

from skimage.morphology import skeletonize as skl


def main():
    path = 'test_img_letter.jpg'
    img = cv2.imread(path, 0)
    # Some smoothing to get rid of the noise
    # img = cv2.bilateralFilter(img, 5, 35, 10)
    img = cv2.GaussianBlur(img, (3, 3), 3)
    # img = cv2.blur(img, (3, 3))
    img = resize(img, width=700)
    # Preprocessing to get the shapes
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 11)
    # Invert to highlight the shape
    th = cv2.bitwise_not(th)
    kernel = np.array([[0, 1, 1],
                       [0, 1, 0],
                       [1, 1, 0]], dtype='uint8')
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('mask', th)
    # cv2.waitKey(0)
    # Skeletonize the shapes
    # Skimage function takes image with either True, False or 0,1
    # and returns and image with values 0, 1.
    th = th == 255
    th = skl(th)
    th = th.astype(np.uint8) * 255
    # Find contours of the skeletons
    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Sort the contours left-to-right
    contours, _ = sort_contours(contours, "left-to-right")

    # List for endpoints
    endpoints = []
    # List of jointpoints
    jointpoints = []
    # List for (x, y) coordinates of the skeletons
    letter_skeletons = []

    for contour in contours:
        if cv2.arcLength(contour, True) > 100:
            # Initialize mask
            mask = np.zeros(img.shape, np.uint8)
            # Bounding rect of the contour
            x, y, w, h = cv2.boundingRect(contour)
            mask[y:y + h, x:x + w] = 255
            # Get only the skeleton in the mask area
            mask = cv2.bitwise_and(mask, th)
            # Take the coordinates of the skeleton points
            rows, cols = np.where(mask == 255)
            # Add the coordinates to the list
            letter_skeletons.append(list(zip(cols, rows)))

            # Find the endpoints for the shape and update a list
            eps = skeleton_endpoints(mask)
            endpoints.append(eps)

            # Find the jointpoints for the shape and update a list
            jps = skeleton_jointpoints(mask)
            jointpoints.append(jps)

            # Draw the endpoints
            [cv2.circle(mask, ep, 5, 255, 1) for ep in eps]
            print("Endpoints %s" % eps)
            [cv2.circle(mask, jp, 4, 180, 1) for jp in jps]
            print("Jointpoints %s" % jps)
            cv2.imshow('mask', mask)
            cv2.waitKey(0)

    # Stack the original and modified
    th = resize(np.hstack((img, th)), 1000)

    # cv2.imshow('mask', th)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Plotting the coordinates extracted
    import matplotlib.pyplot as plt

    for letter_skeleton in letter_skeletons:
        grayscale_color = 0.3
        for px in letter_skeleton:
            x, y = px
            grayscale_color = (grayscale_color + 0.005) % 1
            plt.plot(x, y, linestyle='solid', color=str(grayscale_color), marker='.')

        plt.gca().invert_yaxis()
        plt.pause(5)
        plt.draw()

    # Try to find all of the edges of the letter, with each edge starting & ending at an endpoint
    possible_points = letter_skeletons[0].copy()
    edges = []
    joint_end_points = endpoints[0] + jointpoints[0]

    formats = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
               '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
               '#000075', '#808080', '#ffffff', '#000000']

    img = np.zeros_like(mask)

    new_points = True
    size = 10
    # Loop until new edges are no longer found from the endpoints
    while new_points:
        new_points = False
        size = size + 5
        for index, path_start_point in enumerate(joint_end_points):
            # find_edge will modify possible_points, so we don't end up creating
            # duplicate edges with the same coordinates
            cv2.circle(img, path_start_point, size, 255, 1)
            [cv2.circle(img, point, 3, 150, 1) for point in joint_end_points]
            cv2.imshow('mask', img)
            edge = find_edge_path(path_start_point, joint_end_points.copy(), possible_points)
            if edge is not None:
                new_points = True
                edges.append(edge)
                for point in edge:
                    img[point[1], point[0]] = 255
                cv2.imshow('mask', img)
                cv2.waitKey(0)
                # region Code for displaying lines individually with matplotlib
                # x = []
                # y = []
                # for point in edge:
                #     x.append(point[0])
                #     y.append(point[1])
                #
                # plt.plot(x, y, formats[index % len(formats)])
                # plt.gca().invert_yaxis()
                # plt.pause(2)
                # plt.draw()
                # endregion

    # TODO: Maybe Look at all the edges, and merge the really short ones (<5 long),
    #  the 5 should be the stroke width.
    #  These exist because some jointpoints are too close together.
    # merge_short_edges(edges)

    print("Number of edges: %s" % len(edges))

    for index, edge in enumerate(edges):
        x = []
        y = []
        for point in edge:
            x.append(point[0])
            y.append(point[1])

        plt.plot(x, y, formats[index % len(formats)], antialiased=False, marker='.', ms=0.1)

    plt.gca().invert_yaxis()
    plt.show()


def merge_short_edges(e):
    edges = e.copy()
    for i in range(len(edges)):
        for j in range(len(edges)):
            if len(edges[i]) < 5 or len(edges[j]) < 5:
                print("trying to merge a short edge")


def find_edge_path(start_point, possible_endpoints, possible_points):
    edge = [start_point]
    # Remove the current coordinate (start_point) from the list of possibles, to avoid looping back on itself
    if start_point in possible_points:
        possible_points.remove(start_point)
    if start_point in possible_endpoints:
        possible_endpoints.remove(start_point)

    current_coordinate = start_point

    while True:
        current_coordinate = find_next_coordinate(current_coordinate, possible_points, possible_endpoints)

        if current_coordinate in possible_endpoints:
            edge.append(current_coordinate)
            return edge

        if current_coordinate is not None:
            edge.append(current_coordinate)
            possible_points.remove(current_coordinate)
        else:
            return None


def find_next_coordinate(current_coordinate, possible_coordinates, possible_endpoints):
    # Search around the point to find an adjacent coordinate that is also an endpoint.
    # Otherwise just return the next possible coordinate
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            if x_offset == 0 and y_offset == 0:
                continue
            x, y = current_coordinate
            new_coordinate = (x + x_offset, y + y_offset)
            if new_coordinate in possible_endpoints:
                return new_coordinate

    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            if x_offset == 0 and y_offset == 0:
                continue
            x, y = current_coordinate
            new_coordinate = (x + x_offset, y + y_offset)
            if new_coordinate in possible_coordinates:
                return new_coordinate
    return None


def skeleton_jointpoints(skel):
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 3, 1],
                       [3, 10, 3],
                       [1, 3, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value greater than or equal to 17 (i.e has at least 3 neighbouring pixels,
    # and at least 2 of them are in the horizontal/vertical direction
    # this returns a mask of the jointpoints
    rows, cols = np.where(filtered >= 17)
    coords = list(zip(cols, rows))

    # Remove all jointpoints that are next to each other (i.e. distance < sqrt(2) which we say is ~ 1.5
    for point1 in coords:
        for point2 in coords:
            if point2 not in coords or point1 == point2:
                continue
            x1, y1 = point1
            x2, y2 = point2
            # Pythagoras to figure out distance
            distance = sqrt((abs(x1 - x2)) ** 2 + (abs(y1 - y2)) ** 2)
            if distance < 1.5:
                print("Joint Points %s and %s are very close (%s), removing %s" % (point1, point2, distance, point2))
                coords.remove(point2)

    return coords


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


if __name__ == "__main__":
    main()
