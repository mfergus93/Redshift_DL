import pandas as pd
import numpy as np
import os
import gc
import time
from urllib.error import URLError

from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy import units as u
from astropy.utils.data import conf

conf.cache = False

def get_ugriz_images(ra, dec, radius=0.05, retries=2, delay=5):
    
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
                    # imgshape = images[band].shape
                else:
                    print(f"Failed to retrieve {band}-band image.")
                break  # Exit the retry loop if successful
            except (URLError, ConnectionError) as e:
                print(f"Error retrieving {band}-band image: {e}. Retrying...")
                attempt += 1
                time.sleep(delay*attempt)  # Wait before retrying
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        else:
            print(f"Failed to retrieve {band}-band after {retries} retries, skipping this image.")
            return None
        
    return images if images else None



def save_as_npz(images, filename):
    np.savez_compressed(filename, **images)

galaxies=pd.read_csv('galaxy.csv')

# Get ugriz images
path='D:/galactic_images_ugriz/'

BATCH_SIZE=100
images_batch={}

c=1
batch_number=22
start_idx = (batch_number-1) * BATCH_SIZE #remove the 2 later when batch size back to 100
c = start_idx+1

fail_list=[]

for idx, row in enumerate(galaxies.itertuples(index=False)):
    if idx < start_idx:
        continue
    
    ra=row.ra
    dec=row.dec
    
    images = get_ugriz_images(ra, dec)
    if images is None:
        print(f"Skipping galaxy at c={c} due to repeated failures.")
        fail_list.append(idx)
        continue  # Skip this galaxy
        
    print('ugriz get success ', c)
    event = row.specobjid
    images_batch[f"{event}"] = images
    
    if c % BATCH_SIZE == 0:
        print('batch saving ', batch_number)
        save_as_npz(images_batch, os.path.join(path, f"batch_{batch_number}.npz"))
        print('batch saved', batch_number)
        images_batch.clear()
        batch_number+=1
        gc.collect()
        print('batch success', batch_number)
    c += 1
    SDSS.clear_cache()
    
    if batch_number>1000:
        break
        
if images_batch:
    save_as_npz(images_batch, os.path.join(path, 'final_batch.npz'))
        



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