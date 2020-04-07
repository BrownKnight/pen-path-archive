import neural_network.uji_encoder as uji_encoder

print("Creating GT Files")
uji_encoder.main()

print("Processing image_input")
import image_processing.main as image_processing

image_processing.run("directory")

print("Filtering out gt files")
import neural_network.filter_gt_files
