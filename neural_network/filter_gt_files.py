"""
Look at the image_output directory and remove any files from the ground_truth and
image_input files that are not present in the image_output
"""
import glob
from pathlib import Path

image_output_file_paths = sorted(glob.glob("test.nosync/image_output/*"))
image_output_file_names = [Path(name).stem for name in image_output_file_paths]
print("Found %s image_output files" % len(image_output_file_names))

# Look at every file in the ground truth folder,
# and remove it if the corresponding file doesn't exist in image_output
ground_truth_files = sorted(glob.glob("test.nosync/ground_truth/*"))
print("Found %s ground_truth files" % len(ground_truth_files))
for file_path in ground_truth_files:
    file = Path(file_path)
    if file.stem not in image_output_file_names:
        print("Deleting %s" % file.stem)
        file.unlink()
        Path("test.nosync/image_input/%s.tif" % file.stem).unlink()

