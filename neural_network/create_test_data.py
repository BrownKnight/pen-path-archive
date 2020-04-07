from neural_network import uji_encoder

print("Creating GT Files")
uji_encoder.main()

print("Processing image_input")
from image_processing import main as image_processing

image_processing.run("directory")

print("Filtering out gt files")
from neural_network import filter_gt_files
