import numpy as np
import matplotlib.pyplot as plt
import os

from astropy.visualization import ZScaleInterval

path='D:/galactic_images_ugriz/'
npz_file_path = os.path.join(path,'batch_1.npz')
data = np.load(npz_file_path, allow_pickle=False)

# Store each array in memory
arrays = {key: data[key] for key in data.files}

for key, array in arrays.items():
    print(f"Array '{key}': Shape: {array.shape}, Data Type: {array.dtype}")

# Number of images to display
num_images_to_display = 1  # Adjust this number as needed

# # Create a figure with subplots
# fig, axes = plt.subplots(1, num_images_to_display, figsize=(15, 5))

# # Display each image
# for i, (key, array) in enumerate(arrays.items()):
#     if i >= num_images_to_display:
#         break
#     axes[i].imshow(array, cmap='gray')  # Use cmap='gray' for grayscale images
#     axes[i].set_title(f"{key}")  # Title for each subplot
#     axes[i].axis('off')  # Turn off axis labels

# plt.tight_layout()
# plt.show()

key0=list(arrays.keys())[0]
images=arrays[key0]

def plot_images(images):
    zscale = ZScaleInterval()
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i, band in enumerate(['u', 'g', 'r', 'i', 'z']):
        if band in images:
            img = images[band]
            zmin, zmax = zscale.get_limits(img)
            axes[i].imshow(img, origin='lower', cmap='gray', vmin=zmin, vmax=zmax)
            axes[i].set_title(f'{band}-band')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

plot_images(images)