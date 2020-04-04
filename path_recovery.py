"""
To run this file, use no arguments for default parameters, or use the following syntax
python3 path_recovery.py <mode:single|directory> <input_image_path>

Runs the whole path recovery program by executing the following steps:
- Load the image to be analysed
- Run the image_processing package against it
- Run the output from image_processing in neural_network
"""
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import cv2
import sys
import numpy as np

import image_data_utils
import image_processing.main as image_processing
import neural_network.network as neural_network

MODEL_PATH = "models/bi-lstm-s2s-all_data_w_rotation-epoch_12300e.h5"


def main(model, image_path, working_directory):
    file_name = image_path.stem

    image_input_path = str(image_path)
    image_output_path = "%s/image_output/%s.csv" % (working_directory, file_name)
    prediction_image_output = "%s/prediction_image/%s.tiff" % (working_directory, file_name)
    prediction_path_output = "%s/prediction_path/%s.csv" % (working_directory, file_name)
    analysis_image_output = "%s/analysis_image/%s.tiff" % (working_directory, file_name)

    # Extract undirected edges from the image
    print("Extracting edges from image")
    image_processing.main(image_input_path, image_output_path)

    # Run the undirected image through the neural network to extract pen path
    print("Running edges through Neural Network")
    input_image_data = image_data_utils.load_x(image_output_path)
    predicted_path = neural_network.predict(model, input_image_data.copy())

    print("Saving results")
    # Save the prediction results to various data and image files
    np.savetxt(prediction_path_output, predicted_path.astype(np.uint8), delimiter=",", fmt="%d")
    predicted_image = image_data_utils.create_image_from_data(predicted_path)
    cv2.imwrite(prediction_image_output, predicted_image.astype(np.uint8))

    # Display the images for visual evaluation
    fig: Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colorbar1 = ax1.imshow(image_data_utils.create_image_from_data(input_image_data[0]))
    colorbar2 = ax2.imshow(predicted_image)

    fig.colorbar(colorbar1, ax=ax1)
    fig.colorbar(colorbar2, ax=ax2)
    fig.suptitle(image_input_path)
    fig.tight_layout()

    plt.show()
    plt.savefig(analysis_image_output)


def create_dirs(working_directory):
    # Input path should be in the format "*/<working_directory>/image_input/<file_name>.tif"
    image_output_path = "%s/image_output" % working_directory
    prediction_image_output = "%s/prediction_image" % working_directory
    prediction_path_output = "%s/prediction_path" % working_directory
    analysis_image_output = "%s/analysis_image" % working_directory
    # Create the directories if they don't already exist
    Path(image_output_path).mkdir(parents=True, exist_ok=True)
    Path(prediction_image_output).mkdir(parents=True, exist_ok=True)
    Path(prediction_path_output).mkdir(parents=True, exist_ok=True)
    Path(analysis_image_output).mkdir(parents=True, exist_ok=True)

    # Print some information about this run
    print("Using Model: %s" % MODEL_PATH)
    print("Image Output Path: %s" % image_output_path)
    print("Prediction Output Path: %s" % prediction_image_output)
    print("Prediction Path Output Path: %s" % prediction_path_output)
    print("Analysis Output Path: %s" % analysis_image_output)


if __name__ == "__main__":
    args = sys.argv

    print("Loading model")
    lstm_model = neural_network.load_model(MODEL_PATH)

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

            for file in sorted(path.glob("*")):
                print("------------------Processing file %s------------------" % file)
                main(lstm_model, file, root_dir)

        else:
            exit("Other modes not yet supported, only Single or Directory")

    else:
        input_path = Path("test/image_input/e.tif")
        main(input_path, input_path.parent.parent)

