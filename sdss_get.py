# https://skyserver.sdss.org/dr16/en/help/docs/api.aspx

import pandas as pd
import os
import numpy as np
import requests
import time


#Get img cutout from DR14, requires ra/dec/width
def save_image_from_api(ra, dec, width, height, opt, path='./images/', image_id='image_001', retries=3, delay=5):
    """
    Fetches an image from the specified API URL based on the given parameters and saves it to a file.
    Retries multiple times in case of failure.

    Parameters:
    - ra: Right ascension coordinate.
    - dec: Declination coordinate.
    - width: Width of the image.
    - height: Height of the image.
    - opt: Options for the image.
    - path: Path to save the image files. Defaults to './images/'.
    - image_id: Unique identifier for the image file. Defaults to 'image_001'.
    - retries: Number of times to retry in case of failure.
    - delay: Initial delay between retries in seconds. Will double after each retry.
    """

    # API URL
    api_url = 'https://skyserver.sdss.org/dr14/SkyServerWS/ImgCutout/getjpeg'
    

    # Parameters
    params = {
        'ra': ra,
        'dec': dec,
        'width': width,
        'height': height,
        'opt': opt
    }

    # Create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    attempt = 0
    while attempt < retries:
        try:

            # Sending the request
            response = requests.get(api_url, params=params, timeout=15)
        
            # Checking if the request was successful (status code 200)
            if response.status_code == 200:
                # Save the image
                image_path = os.path.join(path, f'{image_id}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                return True
                # print("Image saved successfully at:", image_path)
            else:
                print("Failed to retrieve the image. Status code:", response.status_code)
        except requests.exceptions.Timeout:
            print(f"Request timed out for image {image_id}. Retrying...")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for image {image_id}: {e}. Retrying...")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Retrying...")
            
        # Increment attempt and wait before retrying (exponential backoff)
        attempt += 1
        time.sleep(delay * (2 ** (attempt - 1)))
    
    # If all retries fail
    print(f"Failed to retrieve image {image_id} after {retries} attempts.")
    return False

#Load
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)
df = pd.read_csv(r'galaxy.csv')

# List to store indices of failed attempts
failed_indices = []

#operator index
# for index, row in df.iterrows():
# for index, row in df.head(100).iterrows():

resume_index=176175
for index, row in df.iloc[resume_index:].iterrows():

    ra = row['ra']
    dec = row['dec']
    phot_id = row['specobjid']
    path='D:/galactic_images_production/'
    
    # print(ra, dec, phot_id, path)
    success = save_image_from_api(ra, dec, 512, 512,'',path, phot_id)
    
    if not success:
        failed_indices.append(index)
        
    if index%100==0:
        print(index)
    
