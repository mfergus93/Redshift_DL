#Here we validate that all our ugriz functions work
import pandas as pd
import random
from ugriz_functions import get_ugriz_images, save_as_hdf5, load_from_hdf5, get_total_directory_size, batch_and_save
from ugriz_functions import check_specobjid_occurrences, filter_incomplete_bands, get_ugriz_HDU_images
import os
from astroquery.sdss import SDSS
from astropy import coordinates as coords

from astropy.io import fits


test_path = 'D:/galactic_images_ugriz_test/'
galaxies = pd.read_csv('galaxy.csv')

randint=random.randint(0, 200_000)

#test if we can get a UGRIZ image from SDSS
#30134 is an index that returned an error?
def test_get_image():
    ra = galaxies.ra[randint]
    dec = galaxies.dec[randint]
    specobjid = galaxies.specobjid[randint]
    image = get_ugriz_images(ra, dec, specobjid)
    
    return image

#test if we can save batches
def test_batch_save():
    galaxies_test_batch=galaxies[0:1000]
    batch_and_save(galaxies_test_batch, batch_size=100, max_batches=2, start_batch_number=2)

# test if we can load a batch or batches
def test_batch_load():
    for i in range(1,11):
        test_batch_path=os.path.join(test_path,'ugriz_images_batch_302.h5')
        images=load_from_hdf5(test_batch_path)
        
        # #test to see what how an error appears in a batch
        for img in images:
            shape=img[2].shape
            if shape!=(1489,2048):
                print(img[2].shape)

#we found out that an error is represented in the batch having missing bands
def test_batch_error_filters(images):
    check=check_specobjid_occurrences(images)
    filtered_images = filter_incomplete_bands(images)
    return check, filtered_images

ra = galaxies.ra[randint]
dec = galaxies.dec[randint]
specobjid = galaxies.specobjid[randint]
test_HDU=get_ugriz_HDU_images(ra,dec,specobjid, radius=0.005)


# Open a FITS file
hdul=test_HDU[0][2]
hdul=hdul[0]

# Access Primary HDU
primary_hdu = hdul[0]
primary_data = primary_hdu.data
primary_header = primary_hdu.header

# Access Image HDU
image_hdu = hdul[1]  # This assumes the ImageHDU is the second HDU
image_data = image_hdu.data
image_header = image_hdu.header

# Access BinTable HDUs
table_hdu1 = hdul[2]  # First binary table HDU
table_data1 = table_hdu1.data

table_hdu2 = hdul[3]  # Second binary table HDU
table_data2 = table_hdu2.data

# You can now inspect the data and headers
print(primary_header)
print(image_header)
print(table_data1)
print(table_data2)



# sky_coords = coords.SkyCoord(ra, dec, unit="deg")
# img = SDSS.get_images()
# test1 = SDSS.query_crossid_async(ra,dec)
# test2 = SDSS.query_region_async(ra,dec)
# test3 = SDSS.query_specobj_async(ra,dec)
# test4 = SDSS.query_photoobj_async(ra,dec)
# test5 = SDSS.query_sql_async(ra,dec)
