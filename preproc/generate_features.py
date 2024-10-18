#main routine
import os
from write_features import preprocess_directory, write_features

folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)
features_dir = os.path.join(folder_path, '..', 'features')

# Ensure the directory exists
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

#creates experimental sets of features
image_dir=r'D:\galactic_images_production'

image_sizes = [512, 256, 128, 64, 32]
color_spaces = ['rgb', 'hsv']
flood_fill_flags = [False, True]

for size in image_sizes:
    for color in color_spaces:
        for flood_fill_flag in flood_fill_flags:
            # Preprocess the directory
            features = preprocess_directory(image_dir, color, flood_fill_flag, size)
            
            # Determine filename based on parameters
            ff_status = 'ff' if flood_fill_flag else 'noff'
            output_filename = os.path.join(features_dir, f"features_{ff_status}_{color}_{size}.pkl")
            
            # Write the features to file
            write_features(features, output_filename)
            
            # Clear the variable to free up memory
            del features
