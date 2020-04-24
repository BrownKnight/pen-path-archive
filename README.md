# pen-path
Digital pen Path Recovery From Handwriting

## Directory/Script Structure
### `/path_recovery.py`

The main entry point for the whole system. This script will take a single or a directory of input images, extract the 
undirected edges from them using the `image_processing/main.py` script, then run those edges through the trained model 
with the `neural_network/network.py` script to direct the edges. 

All of this is done in the `/test/` directory, where many additional directories will be created documenting each 
step in detail. The script requires at least one image in the `/test/image_input/` directory to run.

Syntax: `python3 path_recovery.py <mode:single|directory> <input_image_path>`

### `/test/*`

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


### `/image_processing/*`

Contains all scripts required to extract undirected edges from an input image. Aside from the `main.py` and 
`seperate_chars.py` scripts, all the scripts are not intended to be run directly and are simply used to divide up the 
structure of the program, and are instead called from the `main.py` script.

#### `/image_processing/main.py`

The main entry point for the image processing portion of the system. The script can be run in 2 different modes, 
`single` where the script only processes a specified file, or `directory` where the script process a whole directory of 
files.

Run with three parameters depending on situation
`python3 image_processing/main.py <mode:single|directory> <input_path> <output_path>` 
Where the input_path and output_path are paths to a file or a directory depending on what mode the script is run in.

#### `/image_processing/seperate_chars.py`

This script uses OpenCVs contours and bounding boxes to create a new 64x64px image for every character in the image 
supplied to it.

Syntax: `python3 image_processing/seperate_chars <input_file_path> <output_directory>`

### `/neural_network/*`

This directory contains all the scripts required to both train and use the neural network including creating the 
training data.

### `/neural_network/network.py`

The main entry point for using the neural network. Intended to be used from another script if predicting but can be 
called directly for training. All training data must be in the `/test.nosync/` directory. This directory should contain
3 different directories: `ground_truth` containing the ground truth pen paths; `image_output` containing the undirected
edge output from the image_processing scripts

Syntax: `python3 neural_network/network.py`

### `/neural_network/uji_encoder.py`

Using the UJIpenchars dataset in the `/orignal_data/` directory, this script creates the training data for the neural 
network. It also employs some data augmentation techniques to create up to 200x the samples from the orignal dataset, 
by using various scaling, offset and rotation factors. It will populate the `/test.nosync/image_input` and 
`/test.nosync/ground_truth` directories. The `image_processing/main.py` script should be run on the 
`/test.nosync/image_input` directory to populate the `/test.nosync/image_output` directory required for training.

### `neural_network/filter_gt_files.py`

This script is run when the `/image_processing/main.py` script is not able to extract edges for every file in the 
`/test.nosync/image_input` directory. It will remove any character from the `/test.nosync/ground_truth` directory that 
does not have a corresponding file in the `/test.nosync/image_output` directory. This ensures the x and y datasets for 
the neural networks training is of the same size and match with eachother
