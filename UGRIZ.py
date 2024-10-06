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


# Function to download 5-channel ugriz images for a given RA and Dec
# def get_ugriz_images(ra, dec):
#     # Coordinates of the object
#     sky_coords = coords.SkyCoord(ra, dec, unit="deg")
    
#     # Querying the image cutouts for u, g, r, i, z filters
#     bands = ['u', 'g', 'r', 'i', 'z']
#     images = {}
    
#     for band in bands:
#         # Downloading the image
#         img = SDSS.get_images(coordinates=sky_coords, band=band)
#         if img:
#             images[band] = img[0][0].data  # Extract image data
#         else:
#             print(f"Failed to retrieve {band}-band image.")
    
#     return images


def get_ugriz_images(ra, dec, radius=0.05, retries=3, delay=2):
    
    sky_coords = coords.SkyCoord(ra, dec, unit="deg")
    bands = ['u', 'g', 'r', 'i', 'z']
    images = {}

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

galaxies=pd.read_csv('galaxy.csv')

# Get ugriz images
path='D:/galactic_images_ugriz/'

BATCH_SIZE=100
images_batch={}
c=1
batch_number=1

for row in galaxies.itertuples(index=False):
    ra=row.ra
    dec=row.dec
    
    images = get_ugriz_images(ra, dec, radius=0.01408)
    event=row.specobjid
    images_batch[f"{event}"] = images
    
    if c % BATCH_SIZE == 0:
        save_as_npz(images_batch, os.path.join(path, f"batch_{batch_number}.npz"))
        images_batch.clear()
        batch_number+=1
        
    c+=1
        
if images_batch:
    save_as_npz(images_batch, os.path.join(path, 'final_batch.npz'))
        
    
    










# # Plot the images using matplotlib
# zscale = ZScaleInterval()
# fig, axes = plt.subplots(1, 5, figsize=(15, 5))

# for i, band in enumerate(['u', 'g', 'r', 'i', 'z']):
#     if band in ugriz_images:
#         img = ugriz_images[band]
#         zmin, zmax = zscale.get_limits(img)
#         axes[i].imshow(img, origin='lower', cmap='gray', vmin=zmin, vmax=zmax)
#         axes[i].set_title(f'{band}-band')
#         axes[i].axis('off')

# plt.tight_layout()
# plt.show()