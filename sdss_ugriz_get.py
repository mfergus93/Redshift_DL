import os
import requests
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

# Set the directory to file location, spyder quirk?
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

# Load the CSV file
csv_file = 'galaxy2.csv'
df = pd.read_csv(csv_file)

# Output directory for images and FITS files
output_dir = 'sdss_images'
os.makedirs(output_dir, exist_ok=True)

# Define the bands
bands = ['u', 'g', 'r', 'i', 'z']

# Function to download and process FITS files for a galaxy
def download_and_process_fits(row):
    ra = row['ra']
    dec = row['dec']
    specobjid = row['specobjid']
    run = row['run'] if 'run' in row else None
    camcol = row['camcol'] if 'camcol' in row else None
    field = row['field'] if 'field' in row else None
    
    for band in bands:
        fits_url = f"https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/{run}/{camcol}/frame-{band}-{run}-{camcol}-{field}.fits.bz2"
        response = requests.get(fits_url)
        
        if response.status_code == 200:
            fits_filename = f"{output_dir}/frame-{band}-{run}-{camcol}-{field}.fits.bz2"
            with open(fits_filename, 'wb') as f:
                f.write(response.content)
            
            # Decompress the FITS file
            os.system(f"bunzip2 {fits_filename}")
            decompressed_filename = fits_filename[:-4]
            
            # Read and process the FITS file
            with fits.open(decompressed_filename) as hdul:
                image_data = hdul[0].data
                
            # Plot and save the image
            plt.imshow(image_data, cmap='gray', origin='lower')
            plt.colorbar()
            plt.title(f"Galaxy {specobjid} - Band {band}")
            plt.savefig(f"{output_dir}/image_{specobjid}_{band}.png")
            plt.close()
        else:
            print(f"Failed to download {fits_url}")

# Iterate over the rows of the DataFrame and process each galaxy
for _, row in df.iterrows():
    download_and_process_fits(row)

print("Download and processing complete.")

# for galaxy in galaxies:
#     run = galaxy['run']
#     camcol = galaxy['camcol']
#     field = galaxy['field']
    
#     for band in bands:
#         url = f"https://dr12.sdss.org/sas/dr12/boss/photoObj/frames/{run}/{camcol}/frame-{band}-{run}-{camcol}-{field}.fits.bz2"
#         response = requests.get(url)
        
#         if response.status_code == 200:
#             file_name = f"{output_dir}/frame-{band}-{run}-{camcol}-{field}.fits.bz2"
#             with open(file_name, 'wb') as f:
#                 f.write(response.content)
                
#             # Decompress and process the FITS file
#             os.system(f"bunzip2 {file_name}")
#             fits_file = file_name[:-4]  # Remove .bz2 extension
            
#             # Read the FITS file
#             with fits.open(fits_file) as hdul:
#                 image_data = hdul[0].data
            
#             # Plot and save the image
#             plt.imshow(image_data, cmap='gray', origin='lower')
#             plt.colorbar()
#             plt.title(f"Galaxy {galaxy['specObjID']} - Band {band}")
#             plt.savefig(f"{output_dir}/image_{galaxy['specObjID']}_{band}.png")
#             plt.close()
#         else:
#             print(f"Failed to download {url}")

# print("Download and processing complete.")