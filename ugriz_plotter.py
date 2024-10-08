import pandas as pd
import numpy as np
import os

import time
from urllib.error import URLError

from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy import units as u

SDSS.clear_cache()

def get_ugriz_images(ra, dec, radius=0.05, retries=3, delay=2):
    
    sky_coords = coords.SkyCoord(ra, dec, unit="deg")
    bands = ['u', 'g', 'r', 'i', 'z']
    images = {}
    
    # print(radius)

    for band in bands:
        attempt = 0
        while attempt < retries:
            try:
                img = SDSS.get_images(coordinates=sky_coords, band=band, radius=radius*u.deg)
                if img:
                    images[band] = img[0][0].data
                    imgshape=images[band].shape

                else:
                    print(f"Failed to retrieve {band}-band image.")
                break  # Exit the retry loop if successful
            except URLError as e:
                print(f"Error retrieving {band}-band image: {e}. Retrying...")
                attempt += 1
                time.sleep(delay)  # Wait before retrying

    return images

def save_as_npz(images, filename):
    np.savez_compressed(filename, **images)

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

# Load galaxies data
galaxies = pd.read_csv('galaxy.csv')

# Specify the row index you want to test (e.g., index 839 for c=840)
row_index = 400  # Change this to your desired index
row = galaxies.iloc[row_index]

# Get RA and Dec for the specified row
ra = row.ra
dec = row.dec

# Get ugriz images
images = get_ugriz_images(ra, dec, radius=0.01408)
print(f'ugriz get success for index {row_index}')

# Plot the images
plot_images(images)