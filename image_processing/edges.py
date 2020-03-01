import cv2
from matplotlib import pyplot as plt

from character import Character
from globals import SHOW_STEPS


def extract_edges(char: Character, img):
    plot_formats = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                    '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                    '#000075', '#808080', '#ffffff', '#000000']

    # Try to find all of the edges of the letter, with each edge starting & ending at an endpoint
    possible_points = char.skeleton.copy()
    edges = []
    joint_end_points = char.endpoints + char.jointpoints

    new_points_found = True
    size = 0
    # Loop until new edges are no longer found from the endpoints
    while new_points_found and len(possible_points) > 0:
        new_points_found = False
        size = size + 3
        for index, path_start_point in enumerate(joint_end_points):
            # find_edge_path will modify possible_points, so we don't end up creating
            # duplicate edges with the same coordinates
            cv2.circle(img, path_start_point, size, 255, 1)
            [cv2.circle(img, point, 3, 150, 1) for point in joint_end_points]
            edge = find_edge_path(path_start_point, joint_end_points.copy(), possible_points)
            if edge is not None and len(edge) > 2:
                new_points_found = True

                # Add information about the endpoints to the edge
                # 1=normal point, 2=jointpoint, 3=endpoint
                improved_edge = [(value[0], value[1], 1) for value in edge]
                if edge[0] in char.jointpoints:
                    improved_edge[0] = (improved_edge[0][0], improved_edge[0][1], 2)
                if edge[-1] in char.jointpoints:
                    improved_edge[-1] = (improved_edge[-1][0], improved_edge[-1][1], 2)
                if edge[0] in char.endpoints:
                    improved_edge[0] = (improved_edge[0][0], improved_edge[0][1], 3)
                if edge[-1] in char.endpoints:
                    improved_edge[-1] = (improved_edge[-1][0], improved_edge[-1][1], 3)

                edges.append(improved_edge)
                for point in improved_edge:
                    img[point[1], point[0]] = 255

    if SHOW_STEPS:
        print("Number of edges: %s" % len(edges))
        for index, edge in enumerate(edges):
            x = []
            y = []
            for point in edge:
                x.append(point[0])
                y.append(point[1])

            plt.plot(x, y, plot_formats[index % len(plot_formats)], antialiased=False, marker='.', ms=0.1)

        plt.gca().invert_yaxis()
        plt.show()

    return edges


def find_edge_path(start_point, possible_endpoints, possible_points):
    edge = [start_point]
    # Remove the current coordinate (start_point) from the list of possibles, to avoid looping back on itself
    if start_point in possible_points:
        possible_points.remove(start_point)
    if start_point in possible_endpoints:
        possible_endpoints.remove(start_point)

    current_coordinate = start_point

    while True:
        # Search around the point to find an adjacent coordinate that is also an endpoint.
        next_coordinate = find_next_coordinate(current_coordinate, possible_endpoints)

        # If endpoint could not be found, search for the next coordinate in the edge
        if next_coordinate is None:
            next_coordinate = find_next_coordinate(current_coordinate, possible_points)

        current_coordinate = next_coordinate

        if next_coordinate in possible_endpoints:
            edge.append(next_coordinate)
            return edge

        if next_coordinate is not None:
            edge.append(next_coordinate)
            possible_points.remove(next_coordinate)
        else:
            return None


def find_next_coordinate(current_coordinate, possible_coordinates):
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            if x_offset == 0 and y_offset == 0:
                continue
            x, y = current_coordinate
            new_coordinate = (x + x_offset, y + y_offset)
            if new_coordinate in possible_coordinates:
                return new_coordinate
    return None
