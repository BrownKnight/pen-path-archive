"""
Runs the whole path recovery program by executing the following steps:
- Load the image to be analysed
- Run the image_processing package against it
- Run the output from image_processing in neural_network
"""

from matplotlib import pyplot as plt

import image_processing.main as image_processing
import neural_network.network as neural_network


def main():
    # Load data
    file_name = "d"
    image_input_path = "test/%s.tiff" % file_name
    image_output_path = "test/%s.csv" % file_name

    model_path = "models/bi-lstm-300_epoch-all_data.h5"

    # Print some information about this runm
    print("Image Input Path: %s" % image_input_path)
    print("Image Output Path: %s" % image_output_path)
    print("Using model from %s" % model_path)

    # Extract undirected edges from the image
    print("Extracting edges from image")
    image_processing.main(image_input_path, image_output_path)

    # Run the undirected image through the neural network to extract pen path
    print("Running edges through Neural Network")
    result_image = neural_network.predict(model_path, image_output_path)

    # Display the images for visual evaluation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colorbar1 = ax1.imshow(neural_network.create_image_from_data(neural_network.load_x(image_output_path)[0]))
    colorbar2 = ax2.imshow(result_image)

    fig.colorbar(colorbar1, ax=ax1)
    fig.colorbar(colorbar2, ax=ax2)

    plt.show()


if __name__ == "__main__":
    main()
