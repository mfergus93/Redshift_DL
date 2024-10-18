import pandas as pd
import numpy as np
import os
import gc
import time
from urllib.error import URLError
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.utils.data import conf
import h5py

conf.cache = False

def get_ugriz_images(ra, dec, galaxy_id, radius=0.05, retries=2, delay=5):
    sky_coords = coords.SkyCoord(ra, dec, unit="deg")
    bands = ['u', 'g', 'r', 'i', 'z']
    images = []
    
    for band in bands:
        attempt = 0
        while attempt < retries:
            try:
                img = SDSS.get_images(coordinates=sky_coords, band=band, radius=radius * u.deg)
                if img:
                    images.append([galaxy_id, band, img[0][0].data])
                else:
                    print(f"Failed to retrieve {band}-band image.")
                break
            except (URLError, ConnectionError) as e:
                print(f"Error retrieving {band}-band image: {e}. Retrying...")
                attempt += 1
                time.sleep(delay * attempt)
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        else:
            print(f"Failed to retrieve {band}-band after {retries} retries, skipping this image.")
            return None
    
    return images if images else None

# def save_as_hdf5(images, filename):
#     with h5py.File(filename, 'w') as h5f:
#         for key, value in images.items():
#             h5f.create_dataset(key, data=value)

# def load_from_hdf5(filename):
#     with h5py.File(filename, 'r') as h5f:
#         return {key: h5f[key][()] for key in h5f}

def save_as_hdf5(images, filename):
    with h5py.File(filename, 'w') as h5f:
        for galaxy_id, band, image_array in images:
            # Create a unique dataset name for each entry
            dataset_name = f"{galaxy_id}_{band}"
            h5f.create_dataset(dataset_name, data=image_array, compression='gzip')

def load_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5f:
        images = []
        for dataset_name in h5f:
            # Split the dataset name to extract galaxy_id and band
            galaxy_id, band = dataset_name.split('_')
            images.append([galaxy_id, band, h5f[dataset_name][()]])
        return images


def make_batch(galaxies, max_images=10):
    """Create a batch of ugriz images from the given galaxy DataFrame."""
    images_batch = []  # Use a list to store all images
    
    for idx, row in enumerate(galaxies.itertuples(index=False)):
        if idx >= max_images:  # Limit to the first 10 images
            break
        
        ra = row.ra
        dec = row.dec
        images = get_ugriz_images(ra, dec, row.specobjid)
        
        if images is None:
            print(f"Skipping galaxy at index {idx + 1} due to repeated failures.")
            continue  # Skip this galaxy
            
        print(f'ugriz get success for index {idx + 1}')
        images_batch.extend(images)  # Append images to the giant list

        
    return images_batch

def plot_images(images):
    """Plot the ugriz images for the first event in the loaded images."""
    first_event = list(images.keys())[0]
    first_images = images[first_event]
    
    # Create a plot for the first set of images
    plt.figure(figsize=(15, 10))
    for i, band in enumerate(['u', 'g', 'r', 'i', 'z']):
        plt.subplot(1, 5, i + 1)
        plt.imshow(first_images[band], cmap='gray', origin='lower')
        plt.title(f'{band}-band')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# def main():
    
galaxies = pd.read_csv('galaxy.csv')
path = 'D:/galactic_images_ugriz/'
hdf5_filename = os.path.join(path, 'ugriz_images.h5')
npz_filename = os.path.join(path, 'ugriz_images.npz')

# Create a batch of ugriz images
images_batch = make_batch(galaxies, max_images=10)

if images_batch:
    save_as_hdf5(images_batch, hdf5_filename)
    print(f'Batch saved as {hdf5_filename}')

# Load the images from HDF5
loaded_images = load_from_hdf5(hdf5_filename)

# np.savez_compressed(npz_filename, images_batch)

# # Plot the first set of ugriz images
# plot_images(loaded_images)

# if __name__ == '__main__':
#     main()