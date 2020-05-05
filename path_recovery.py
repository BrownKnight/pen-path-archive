"""
To run this file, use no arguments for default parameters, or use the following syntax
python3 path_recovery.py <mode:single|directory> <input_image_path>

Input images should be in the test/image_input directory. They should be 64x64 images with .tif file type
Running image_processing/separate_chars.py will take input from test/multi_char and output to test/image_input

Runs the whole path recovery program by executing the following steps:
- Load the image to be analysed
- Run the image_processing package against it
- Run the output from image_processing in neural_network
"""
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer # Used to measure performance of the system

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import cv2
import sys
import numpy as np

import image_processing.main as image_processing
from image_processing import adopt_path_shape
from neural_network import network, image_data_utils

MODEL_PATH = "models//BiLSTM-S2S.h5"
SAVE_VISUALISATION_IMAGES = False


def main(model, image_path, working_directory):
    file_name = image_path.stem

    image_input_path = str(image_path)
    image_output_path = "%s/image_output/%s.csv" % (working_directory, file_name)
    adopted_path_output = "%s/adopted_path/%s.csv" % (working_directory, file_name)
    adopted_image_output = "%s/adopted_image/%s.tiff" % (working_directory, file_name)
    if SAVE_VISUALISATION_IMAGES:
        prediction_image_output = "%s/prediction_image/%s.tiff" % (working_directory, file_name)
        prediction_path_output = "%s/prediction_path/%s.csv" % (working_directory, file_name)
        analysis_image_output = "%s/analysis_image/%s.tiff" % (working_directory, file_name)

    # Extract undirected edges from the image
    print("Extracting edges from image %s" % image_input_path)
    success = image_processing.main(image_input_path, image_output_path)
    if not success:
        return

    # Run the undirected image through the neural network to extract pen path
    print("Running edges through Neural Network")
    input_image_data = image_data_utils.load_x(image_output_path)
    predicted_path = network.predict(model, input_image_data.copy())

    if SAVE_VISUALISATION_IMAGES:
        print("Saving results")
        # Save the prediction results to various data and image files
        np.savetxt(prediction_path_output, predicted_path.astype(np.uint8), delimiter=",", fmt="%d")
        predicted_image = image_data_utils.create_image_from_data(predicted_path)
        cv2.imwrite(prediction_image_output, predicted_image.astype(np.uint8))

    print("Adopting path to shape of character")
    adopted_path = adopt_path_shape.adopt_path_shape(input_image_data[0], predicted_path)
    adopted_image = image_data_utils.create_image_from_data(adopted_path)
    np.savetxt(adopted_path_output, adopted_path.astype(np.uint8), delimiter=",", fmt="%d")

    cv2.imwrite(adopted_image_output, adopted_image.astype(np.uint8))

    if SAVE_VISUALISATION_IMAGES:
        # Display the images for visual evaluation
        fig: Figure
        ax1: Axes
        ax2: Axes
        ax3: Axes
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        colorbar1 = ax1.imshow(image_data_utils.create_image_from_data(input_image_data[0]))
        ax1.set_title("Edges extracted from Input")
        colorbar2 = ax2.imshow(predicted_image)
        ax2.set_title("Predicted Path")
        colorbar3 = ax3.imshow(adopted_image)
        ax3.set_title("Input Shape Adopted Path")

        fig.colorbar(colorbar1, ax=ax1)
        fig.colorbar(colorbar2, ax=ax2)
        fig.colorbar(colorbar3, ax=ax3)
        fig.suptitle(file_name)
        fig.tight_layout()

        plt.savefig(analysis_image_output)
        plt.show()


def create_dirs(working_directory):
    # Input path should be in the format "*/<working_directory>/image_input/<file_name>.tif"
    image_output_path = "%s/image_output" % working_directory
    adopted_image_output = "%s/adopted_image" % working_directory
    adopted_path_output = "%s/adopted_path" % working_directory

    if SAVE_VISUALISATION_IMAGES:
        prediction_image_output = "%s/prediction_image" % working_directory
        prediction_path_output = "%s/prediction_path" % working_directory
        analysis_image_output = "%s/analysis_image" % working_directory

    # Create the directories if they don't already exist
    Path(image_output_path).mkdir(parents=True, exist_ok=True)
    Path(adopted_image_output).mkdir(parents=True, exist_ok=True)
    Path(adopted_path_output).mkdir(parents=True, exist_ok=True)

    if SAVE_VISUALISATION_IMAGES:
        Path(prediction_image_output).mkdir(parents=True, exist_ok=True)
        Path(prediction_path_output).mkdir(parents=True, exist_ok=True)
        Path(analysis_image_output).mkdir(parents=True, exist_ok=True)

    # Print some information about this run
    print("Using Model: %s" % MODEL_PATH)
    print("Image Output Path: %s" % image_output_path)
    if SAVE_VISUALISATION_IMAGES:
        print("Prediction Output Path: %s" % prediction_image_output)
        print("Prediction Path Output Path: %s" % prediction_path_output)
        print("Analysis Output Path: %s" % analysis_image_output)
    print("Adopted Path Output Path: %s" % adopted_path_output)
    print("Adopted Image Output Path: %s" % adopted_image_output)


if __name__ == "__main__":
    args = sys.argv

    print("Loading model")
    lstm_model = network.load_model(MODEL_PATH)
    print(lstm_model.summary())
    start_time = timer()

    if len(args) == 3:
        mode = args[1]
        path = Path(args[2])

        if mode == "single":
            if not path.exists():
                exit("Inputted path does not exist")

            root_dir = path.parent.parent
            create_dirs(root_dir)
            main(lstm_model, path, root_dir)

        elif mode == "directory":
            if not path.is_dir():
                exit("Inputted path is not a directory")

            root_dir = path.parent
            create_dirs(root_dir)
            file_paths = sorted([path for path in path.glob("*") if not path.stem.startswith(".")])
            for index, file in enumerate(file_paths[:10000]):
                print("---------Processing file #%d (%s)--------" % (index, file.stem))
                main(lstm_model, file, root_dir)

        else:
            exit("Other modes not yet supported, only Single or Directory")

    else:
        exit("Required Syntax: python3 path_recovery.py <mode:single|directory> <input_image_path> For default use python3 path_recovery.py directory test/image_input")

    end_time = timer()
    elapsed_time = end_time - start_time
    print("Elapsed time (hh:mm:ss): %s (%.2f seconds)" % (timedelta(seconds=elapsed_time), elapsed_time))

