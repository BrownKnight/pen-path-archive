"""
To run this file no arguments are required. Ensure the test/image_input and test/adopted_image
folders are correctly populated
python3 evaluation.py

Combines all the images in the image_path to allow for evaluation of the performance of the system.
path_recovery.py should be run before this to populate all the directories
"""
import glob
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from math import floor
import sys
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

import image_data_utils



def process_image_input(image_path):
    file_paths = sorted(glob.glob(image_path + "/*"))
    images: np.ndarray = image_data_utils.load_rgb_images(file_paths)
    # We use different grid sizes for different numbers of images
    if images.shape[0] <= 36:
        fig: Figure
        axes: [Axes]
        fig, axes = plt.subplots(6, 6, figsize=(18, 18))
        plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
        for index, image in enumerate(images):
            axes[floor(index/6), index % 6].imshow(image)
            axes[floor(index/6), index % 6].set_title(Path(file_paths[index]).stem)
            # axes[floor(index/6), index % 6].axis('off')
            axes[floor(index/6), index % 6].get_xaxis().set_ticks([])
            axes[floor(index/6), index % 6].get_yaxis().set_ticks([])
    plt.savefig("test/%s.png" % Path(image_path).stem)
    plt.show()


def process_greyscale_image(image_path):
    file_paths = sorted(glob.glob(image_path + "/*"))
    images: np.ndarray = image_data_utils.load_greyscale_images(file_paths)

    if images.shape[0] <= 36:
        fig: Figure
        # plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
        fig = plt.figure(figsize=(18.5, 18))

        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(6, 6),
                         axes_pad=0.25,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )

        for index, image in enumerate(images):
            colorbar = grid[index].imshow(image)
            grid[index].set_title(Path(file_paths[index]).stem)
            # grid[index].axis('off')
            grid[index].get_xaxis().set_ticks([])
            grid[index].get_yaxis().set_ticks([])

        grid[0].cax.colorbar(colorbar)
        grid[0].cax.toggle_label(True)

    plt.savefig("test/%s.png" % Path(image_path).stem)
    plt.show()


if __name__ == "__main__":
    args = sys.argv
    image_input_path = "test/image_input"
    process_image_input(image_input_path)

    prediction_image_path = "test/prediction_image"
    process_greyscale_image(prediction_image_path)

    adopted_image_path = "test/adopted_image"
    process_greyscale_image(adopted_image_path)
