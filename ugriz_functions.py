# ugriz functions
import pandas as pd
import numpy as np
import os
import gc
import time
from urllib.error import URLError
import h5py
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy import units as u
from PIL import Image

# This function will query SDSS to obtain the UGRIZ images for a ra/dec
# outputs in the format list of [specobjid, band, array]


def get_ugriz_images(ra, dec, galaxy_id, radius=0.05, retries=2, delay=5):
    sky_coords = coords.SkyCoord(ra, dec, unit="deg")
    bands = ['u', 'g', 'r', 'i', 'z']
    images = []

    for band in bands:
        attempt = 0
        while attempt < retries:
            try:
                img = SDSS.get_images(
                    coordinates=sky_coords, band=band, radius=radius * u.deg, cache=False)
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

# saves a list of images as hdf5

def get_ugriz_HDU_images(ra, dec, galaxy_id, radius=0.05, retries=2, delay=5):
    sky_coords = coords.SkyCoord(ra, dec, unit="deg")
    bands = ['u', 'g', 'r', 'i', 'z']
    images = []

    for band in bands:
        attempt = 0
        while attempt < retries:
            try:
                img = SDSS.get_images(
                    coordinates=sky_coords, band=band, radius=radius * u.deg, cache=False)
                if img:
                    images.append([galaxy_id, band, img])
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

def save_as_hdf5(images, filename):
    with h5py.File(filename, 'w') as h5f:
        for galaxy_id, band, image_array in images:
            # Create a unique dataset name for each entry
            dataset_name = f"{galaxy_id}_{band}"
            h5f.create_dataset(
                dataset_name, data=image_array, compression='gzip')


def load_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5f:
        images = []
        for dataset_name in h5f:
            # Split the dataset name to extract galaxy_id and band
            galaxy_id, band = dataset_name.split('_')
            images.append([galaxy_id, band, h5f[dataset_name][()]])
        return images


def get_total_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def batch_and_save(galaxies, batch_size=100, max_size_tb=1, max_batches=None, start_batch_number=1):
    path = 'D:/galactic_images_ugriz/'
    batch_number = 1
    images_batch = []

    # # Calculate starting index based on the starting batch number
    # start_idx = (start_batch_number - 1) * batch_size
    # batch_number = start_batch_number

    for idx, row in galaxies.iterrows():
        ra = row.ra
        dec = row.dec

        images = get_ugriz_images(ra, dec, row.specobjid)
        if images is None:
            print(f"Skipping galaxy at index {idx + 1} due to repeated failures.")
            continue  # Skip this galaxy

        print(f'ugriz get success for index {idx + 1}')
        images_batch.extend(images)  # Append the returned images to the batch

        if len(images_batch) >= batch_size*5:
            # check directory size
            total_size_tb = get_total_directory_size(
                path) / (1024 ** 4)  # Convert bytes to TB
            if total_size_tb >= max_size_tb:
                print(f"Total directory size exceeds {max_size_tb} TB. Stopping batch processing.")
                break

            hdf5_filename = os.path.join(
                path, f'ugriz_images_batch_{batch_number}.h5')
            save_as_hdf5(images_batch, hdf5_filename)
            print(f'Batch {batch_number} saved as {hdf5_filename}')
            images_batch.clear()  # Clear the batch for the next iteration
            batch_number += 1
            gc.collect()  # Force garbage collection

            # Check for maximum number of batches
            if max_batches is not None and batch_number > max_batches:
                print(f"Maximum number of batches {max_batches} reached. Stopping batch processing.")
                break

    # Save any remaining images that didn't fill a complete batch
    if images_batch:
        hdf5_filename = os.path.join(
            path, f'ugriz_images_batch_{batch_number}.h5')
        save_as_hdf5(images_batch, hdf5_filename)
        print(f'Final batch {batch_number} saved as {hdf5_filename}')

def check_specobjid_occurrences(images):
    # Extract all specobjid values
    specobjids = [img[0] for img in images]
    
    # Create a dictionary to store the count of each specobjid
    specobjid_count = {specobjid: specobjids.count(specobjid) for specobjid in set(specobjids)}
    
    # Find specobjids with occurrences not equal to 5
    incorrect_specobjids = {specobjid: count for specobjid, count in specobjid_count.items() if count != 5}
    
    return incorrect_specobjids

# check=check_specobjid_occurrences(images)

def filter_incomplete_bands(images):
    galaxy_band_dict = {}

    # Group images by specobjid
    for img in images:
        specobjid, band, _ = img
        if specobjid not in galaxy_band_dict:
            galaxy_band_dict[specobjid] = set()
        galaxy_band_dict[specobjid].add(band)

    # Find specobjids that are missing bands
    complete_images = []
    for img in images:
        specobjid = img[0]
        if len(galaxy_band_dict[specobjid]) == 5:  # Only keep complete sets of 5 bands
            complete_images.append(img)

    return complete_images

def resize_image(image_array, size=(224, 224)):
    # Convert to uint8 if necessary
    if image_array.dtype == np.float32:
        image_array = (image_array * 255).astype(np.uint8)  # Scale if necessary

    img = Image.fromarray(image_array)
    img_resized = img.resize(size, Image.ANTIALIAS)
    return np.array(img_resized)

def resize_batch(images, size=(224, 224)):
    resized_images = []
    for img in images:
        resized_img = resize_image(img[2], size)  # img[2] is the image array
        resized_images.append([img[0], img[1], resized_img])
    return resized_images

def stack_bands_by_specobjid(images):
    galaxy_images = {}
    
    # Group images by specobjid
    for img in images:
        specobjid, band, array = img
        if specobjid not in galaxy_images:
            galaxy_images[specobjid] = {}
        galaxy_images[specobjid][band] = array
    
    # Stack the bands in the order u, g, r, i, z into a cube
    stacked_images = []
    band_order = ['u', 'g', 'r', 'i', 'z']  # Define the correct band order
    for specobjid, bands in galaxy_images.items():
        image_stack = np.stack([bands[band] for band in band_order], axis=-1)  # Stack along the last axis
        stacked_images.append([specobjid, image_stack])
    
    return stacked_images
