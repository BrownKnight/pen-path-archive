"""A class to hold all the required data for each character"""


class Character:
    """This classes properties are populated throughout all of the image processing steps"""
    """Populated in the character recognition step"""
    letter = ""
    image = []

    """Populated in the skeletonize step"""
    endpoints = []
    jointpoints = []
    skeleton = []

    """Populate in the edge detection/pathing step"""
    edges = []

    @property
    def min_edge_length(self):
        return min([len(i) for i in self.edges] or [0]) or 0

    @property
    def average_edge_length(self):
        return int(sum([len(i) for i in self.edges] or [0]) / (len(self.edges) or 1)) or 0

    @property
    def max_edge_length(self):
        return max([len(i) for i in self.edges] or [0]) or 0

    @property
    def usable(self):
        """Determines if this character has enough information to be used in the Neural Network"""
        return len(self.endpoints) > 0 and self.skeleton is not [] and len(self.edges) > 0
