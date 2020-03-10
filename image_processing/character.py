"""A class to hold all the required data for each character"""
import numpy as np
import cv2
from globals import WAIT_TIME, SHOW_STEPS


class Character:
    """This classes properties are populated throughout all of the image processing steps"""
    image = []

    """Populated in the skeletonize step"""
    endpoints = []
    jointpoints = []
    skeleton = []

    """Populate in the edge detection/pathing step"""
    edges = []

    progress_image: np.ndarray = np.zeros((64, 64, 1))

    def add_to_progress_image(self, img_to_add, caption):
        if SHOW_STEPS:
            img = img_to_add.copy()
            # cv2.putText(img, caption, (0,0), cv2.FONT_HERSHEY_PLAIN, 0.5, 255, 1, bottomLeftOrigin=True)
            self.progress_image = np.hstack((self.progress_image, img))

            cv2.imshow("progress", self.progress_image)
            cv2.waitKey(WAIT_TIME)


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
