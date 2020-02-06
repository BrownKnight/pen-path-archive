"""A class to hold all the required data for each character"""


class Character:
    """This classes properties are populated throughout all of the image processing steps"""
    """Populated in the character recognition step"""
    letter = ""
    image = []

    """Populated in the skeletonize step"""
    endpoints = []
    jointpoints = []

    """Populate in the edge detection/pathing step"""
    edges = []