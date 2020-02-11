import glob
import numpy as np


def load_data(file_dir):
    """Loads all the character data from a given directory and outputs it into a numpy array"""
    file_list = glob.glob(file_dir)
    num_of_files = len(file_list)
    print("Found %s character files" % num_of_files)

    # Create an array of character point arrays
    chars = np.zeros((num_of_files, 128, 3))
    for index, file in enumerate(file_list):
        with open(file) as f:
            lines = f.readlines()
            lines = [line.strip("\n").split(",") for line in lines]
            char = [(line[0], line[1], line[2]) for line in lines]
            chars[index] = char

    return chars


if __name__ == "__main__":
    load_data("test/image_output/*.csv")
