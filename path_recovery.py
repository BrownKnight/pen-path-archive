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
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import cv2
import sys
import numpy as np

import image_data_utils
import image_processing.main as image_processing
from image_processing.adopt_path_shape import adopt_path_shape
import neural_network.network as neural_network

MODEL_PATH = "models/bi-lstm-s2s-all_data_w_rotation-epoch_12800n.h5"


def main(model, image_path, working_directory):
    file_name = image_path.stem

    image_input_path = str(image_path)
    image_output_path = "%s/image_output/%s.csv" % (working_directory, file_name)
    prediction_image_output = "%s/prediction_image/%s.tiff" % (working_directory, file_name)
    prediction_path_output = "%s/prediction_path/%s.csv" % (working_directory, file_name)
    analysis_image_output = "%s/analysis_image/%s.tiff" % (working_directory, file_name)
    adopted_path_output = "%s/adopted_path/%s.csv" % (working_directory, file_name)
    adopted_image_output = "%s/adopted_image/%s.tiff" % (working_directory, file_name)

    # Extract undirected edges from the image
    print("Extracting edges from image")
    success = image_processing.main(image_input_path, image_output_path)
    if not success:
        return

    # Run the undirected image through the neural network to extract pen path
    print("Running edges through Neural Network")
    input_image_data = image_data_utils.load_x(image_output_path)
    predicted_path = neural_network.predict(model, input_image_data.copy())

    print("Saving results")
    # Save the prediction results to various data and image files
    np.savetxt(prediction_path_output, predicted_path.astype(np.uint8), delimiter=",", fmt="%d")
    predicted_image = image_data_utils.create_image_from_data(predicted_path)
    cv2.imwrite(prediction_image_output, predicted_image.astype(np.uint8))

    print("Adopting path to sequence")
    adopted_path = adopt_path_shape(input_image_data[0], predicted_path)
    adopted_image = image_data_utils.create_image_from_data(adopted_path)
    np.savetxt(adopted_path_output, adopted_path.astype(np.uint8), delimiter=",", fmt="%d")

    cv2.imwrite(adopted_image_output, adopted_image.astype(np.uint8))

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
    fig.suptitle(image_input_path)
    fig.tight_layout()

    plt.savefig(analysis_image_output)
    plt.show()


def create_dirs(working_directory):
    # Input path should be in the format "*/<working_directory>/image_input/<file_name>.tif"
    image_output_path = "%s/image_output" % working_directory
    prediction_image_output = "%s/prediction_image" % working_directory
    prediction_path_output = "%s/prediction_path" % working_directory
    analysis_image_output = "%s/analysis_image" % working_directory
    adopted_image_output = "%s/adopted_image" % working_directory
    adopted_path_output = "%s/adopted_path" % working_directory
    # Create the directories if they don't already exist
    Path(image_output_path).mkdir(parents=True, exist_ok=True)
    Path(prediction_image_output).mkdir(parents=True, exist_ok=True)
    Path(prediction_path_output).mkdir(parents=True, exist_ok=True)
    Path(analysis_image_output).mkdir(parents=True, exist_ok=True)
    Path(adopted_image_output).mkdir(parents=True, exist_ok=True)
    Path(adopted_path_output).mkdir(parents=True, exist_ok=True)

    # Print some information about this run
    print("Using Model: %s" % MODEL_PATH)
    print("Image Output Path: %s" % image_output_path)
    print("Prediction Output Path: %s" % prediction_image_output)
    print("Prediction Path Output Path: %s" % prediction_path_output)
    print("Analysis Output Path: %s" % analysis_image_output)
    print("Adopted Image Output Path: %s" % adopted_image_output)


if __name__ == "__main__":
    args = sys.argv

    print("Loading model")
    lstm_model = neural_network.load_model(MODEL_PATH)
    print(lstm_model.summary())

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
        main(lstm_model, input_path, input_path.parent.parent)

