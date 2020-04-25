# pen-path
Digital Pen Path Recovery From Handwriting

The system approaches the problem of extracting pen paths from images of handwritten text using a hybrid approach 
consisting of both classical image processing techniques, and Deep Neural Networks.

The result of this is a system that can take an image of a set of or a single character, and output a directed list of 
points representing the path the pen likely took to draw the character,

## Setup
This is built against Python 3.7. It is recommended to use a venv when running this script for ease of use.
The following packages are dependencies for this project:
- NumPy
- MatPlotLib
- SciKit-Image
- TensorFlow 2.x
- opencv-python (OpenCV2)
- imutils




## Testing/Evaluation the System
### Using the demo images
For the purposes of testing the system and the models, there is a collection of images supplied in the `/demo/` 
directory which can be copied into the `/test/image_input/` directory to allow you to run the `path_recovery.py` 
script. Detailed instructions are below.

1. Empty the `/test/` directory to ensure there are no extraneous results from previous runs. Create this directory if 
    it does not exist already

2. Choose a set of input images from the `/demo/*` directory, and copy them into the a `/test/image_input` directory

3. Run the `/path_recovery.py` script with the parameters `directory test/image_input`

    ```python3 path_recovery.py directory test/image_input```
    
    This will create many new directories in the `/test/` directory, showing each step the system took to produce the 
    output.

4. To help with visualising the results, you may run the `/evaluation.py` script to produce a board of images to allow 
    for easier direct comparison
   
   ```python3 evaluation.py``` 
   
   This will separately compile all images in the `/test/prediction_image`, `/test/adopted_image`, and 
   `/test/image_input` directories, and create an image with the corresponding name for each directory in the 
   `/test/` directory
   
For more information on these scripts, see the [Directory and Script Structure](#directory-and-script-structure) section below
   
### Creating your own images
To evaluate the system against your own handwriting or examples of text, you can simply place an image in the `
/test/multi_char/` directory, then run the `/image_processing/seperate_chars.py` script to extract each character out 
from the image, and create individual character image files in the `/test/image_input/` directory. Then you can simply 
follow the instructions above to extract the pen paths. 

1. Create the `/test/multi_char` directory and place your image containing multiple characters in it.

2. Run the `/image_processing/seperate_chars.py` script to generate the individual character images.
    
    ```python3 image_processing/seperate_chars.py <input_image_path> <output_directory>```
    
    e.g. ```python3 image_processing/seperate_chars.py test/multi_char/all-chars.jpeg test/image_input```
    
    the `<input_image_path>` and `output_directory` parameters can be omitted, in which case the default values 
    `test/multi_char/all-chars.jpeg` and `test/image_input` will be used respectively

3. Follow steps 3 & 4 in the [Using the Demo Images](#Using-the-demo-images) section to extract edges from these images 
you have generated

### Using the UJIpenchars Dataset
This project includes a script in `/neural_network/uji_encoder.py` that can generate training data from the UJIpenchars 
dataset included in the `/original_data/` directory. Follow the instructions in the 
[Directory and Script Structure](#directory-and-script-structure) section below to generate this data which you can 
then use for testing the system.




## Directory and Script Structure
#### `/path_recovery.py`

The main entry point for the whole system. This script will take a single or a directory of input images, extract the 
undirected edges from them using the `image_processing/main.py` script, then run those edges through the trained model 
with the `neural_network/network.py` script to direct the edges. 

All of this is done in the `/test/` directory, where many additional directories will be created documenting each 
step in detail. The script requires at least one image in the `/test/image_input/` directory to run.

Syntax: `python3 path_recovery.py <mode:single|directory> <input_image_path>`

#### `/evaluation.py`

This will separately compile all images in the `/test/prediction_image`, `/test/adopted_image`, and 
`/test/image_input` directories, and create an image with the corresponding name for each directory in the 
`/test/` directory. These paths are all hardcoded.

Syntax: `python3 evaluation.py`

#### `/test/*`

This directory is where any files/directories related to the manual running of the system are held. The 
`path_recovery.py` script will create many directories in here documenting the various steps it takes to produce an 
 output sequence from the input image. These are described below:
- `/test/image_input/` This directory contains all the 64x64 single-character images
- `/test/adopted_path/` This directory contains the directed output sequences for each input image, after they have 
 been run through the neural network and the input shape adoption algorithm.
- `/test/adopted_image/` The `adopted_path` sequences are plotted onto a 64x64px coordinate grid for visualisation 
 purposes
- `/test/image_output/` The undirected list of edges is output here from the `image_processing` portion of the system
- `/test/prediction_path/` The directed pen path output of the neural network, before running through the shape adoption
 algorithm
- `/test/prediction_image/` Every path in the `prediction_path` directory is plotted on a 64x64px coordinate grid to
 allow for easy visual analysis of the networks performance  
- `/test/analysis_image/` The `image_input`, `prediction_image`, and `adopted_image` images for each character are plotted 
next to each other to allow for visualisation of the systems output


#### `/image_processing/*`

Contains all scripts required to extract undirected edges from an input image. Aside from the `main.py` and 
`seperate_chars.py` scripts, all the scripts are not intended to be run directly and are simply used to divide up the 
structure of the program, and are instead called from the `main.py` script.

##### `/image_processing/main.py`

The main entry point for the image processing portion of the system. The script can be run in 2 different modes, 
`single` where the script only processes a specified file, or `directory` where the script process a whole directory of 
files.

Run with three parameters depending on situation
`python3 image_processing/main.py <mode:single|directory> <input_path> <output_path>` 
Where the input_path and output_path are paths to a file or a directory depending on what mode the script is run in.

##### `/image_processing/seperate_chars.py`

This script uses OpenCVs contours and bounding boxes to create a new 64x64px image for every character in the image 
supplied to it.

Syntax: `python3 image_processing/seperate_chars <input_file_path> <output_directory>`

The `<input_image_path>` and `output_directory` parameters can be omitted, in which case the default values 
`test/multi_char/all-chars.jpeg` and `test/image_input` will be used respectively

##### `/image_processing/globals.py`

This script contains just two variables which are used in various `image_processing` scripts. If `SHOW_STEPS` is set to
 True, throughout the various processing steps a OpenCV window will open showing the progress of the script, and the 
 various steps it is taking. Each step progresses after the `WAIT_TIME` time has passed (in milliseconds). If 
 `WAIT_TIME` is set to zero, the script only processes onto the next step after any keyboard input.

#### `/neural_network/*`

This directory contains all the scripts required to both train and use the neural network including creating the 
training data.

#### `/neural_network/network.py`

The main entry point for using the neural network. Intended to be used from another script if predicting but can be 
called directly for training. All training data must be in the `/test.nosync/` directory. This directory should contain
3 different directories: `ground_truth` containing the ground truth pen paths; `image_output` containing the undirected
edge output from the image_processing scripts

Syntax: `python3 neural_network/network.py`

#### `/neural_network/uji_encoder.py`

Using the UJIpenchars dataset in the `/orignal_data/` directory, this script creates the training data for the neural 
network. It also employs some data augmentation techniques to create up to 200x the samples from the original dataset, 
by using various scaling, offset and rotation factors. It will populate the `/test.nosync/image_input` and 
`/test.nosync/ground_truth` directories. The `image_processing/main.py` script should be run on the 
`/test.nosync/image_input` directory to populate the `/test.nosync/image_output` directory required for training.

To change the number of samples generates by the script, simply change the `SCALING_FACTOR_LIST`, `OFFSET_LIST`, and 
`ROTATION_LIST` constants at the top of the script.

#### `/neural_network/filter_gt_files.py`

This script is run when the `/image_processing/main.py` script is not able to extract edges for every file in the 
`/test.nosync/image_input` directory. It will remove any character from the `/test.nosync/ground_truth` directory that 
does not have a corresponding file in the `/test.nosync/image_output` directory. This ensures the x and y datasets for 
the neural networks training is of the same size and match with eachother
