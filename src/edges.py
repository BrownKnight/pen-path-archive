import cv2
from matplotlib import pyplot as plt


def extract_edges(endpoints, jointpoints, letter_skeleton, img):
    plot_formats = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                    '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                    '#000075', '#808080', '#ffffff', '#000000']

    # Try to find all of the edges of the letter, with each edge starting & ending at an endpoint
    possible_points = letter_skeleton.copy()
    edges = []
    joint_end_points = endpoints + jointpoints

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
            if edge is not None:
                new_points_found = True
                edges.append(edge)
                for point in edge:
                    img[point[1], point[0]] = 255

    cv2.imshow('letters', img)
    cv2.waitKey(0)

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
