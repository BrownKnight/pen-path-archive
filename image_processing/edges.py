import cv2
import numpy as np
from matplotlib import pyplot as plt

from character import Character
from globals import SHOW_STEPS


def extract_edges(char: Character):
    plot_formats = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                    '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                    '#000075', '#808080', '#ffffff', '#000000']
    visualisation_image = np.zeros_like(char.image)
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
        for path_start_point in joint_end_points:
            cv2.circle(visualisation_image, path_start_point, size, 200, 1)
            [cv2.circle(visualisation_image, point, 3, 150, 1) for point in joint_end_points]

            # find_edge_path will modify possible_points, so we don't end up creating
            # duplicate edges with the same coordinates
            new_edges = find_edge_path([path_start_point], possible_points, joint_end_points.copy())
            print(new_edges)
            for edge in new_edges:
                if edge is not None and len(edge) > 2:
                    new_points_found = True

                    # Add information about the endpoints to the e
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
                        visualisation_image[point[1], point[0]] = 255

                    # # DEBUG
                    # cv2.namedWindow('debug', flags=cv2.WINDOW_NORMAL)
                    # cv2.imshow('debug', visualisation_image)
                    # cv2.waitKey(0)
                # else:
                #     # # DEBUG
                #     # print("edge failed %s" % edge)

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

    char.add_to_progress_image(visualisation_image, "edges")

    return edges


def find_edge_path(current_edge: list, possible_points, endpoints):
    edges = [current_edge]
    new_points_found = True
    while new_points_found:
        new_points_found = False
        for edge in edges:
            if len(edge) > 1 and edge[-1] in endpoints:
                continue

            next_points = find_adjacent_points(edge, possible_points, endpoints)

            if len(next_points) == 0:
                continue

            new_points_found = True
            # Add the current point to
            edge.append(next_points.pop(0))

            if len(next_points) > 0:
                for point in next_points:
                    edges.append(edge[:-1] + [point])
    return edges


def find_adjacent_points(edge, possible_points, endpoints):
    current_point = edge[-1]
    adjacent_points = []
    for x in range(-1,2,1):
        for y in range(-1,2,1):
            if x == 0 and y == 0:
                continue

            point = (current_point[0] + x, current_point[1] + y)
            if point in possible_points and point not in edge:
                adjacent_points.append(point)
                possible_points.remove(point)
            elif point in endpoints and point not in edge:
                adjacent_points.append(point)

    return adjacent_points