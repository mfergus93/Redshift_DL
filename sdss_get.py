# https://skyserver.sdss.org/dr16/en/help/docs/api.aspx

import pandas as pd
import os
import numpy as np
import requests
import time

folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)
df = pd.read_csv(r'galaxy.csv')

def save_image_from_api(ra, dec, width, height, opt, path='./images/', image_id='image_001'):
    
    """
    Fetches an image from the specified API URL based on the given parameters and saves it to a file.

    Parameters:
    - ra: Right ascension coordinate.
    - dec: Declination coordinate.
    - width: Width of the image.
    - height: Height of the image.
    - opt: Options for the image.
    - path: Path to save the image files. Defaults to './images/'.
    - image_id: Unique identifier for the image file. Defaults to 'image_001'.
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

    # Sending the request
    response = requests.get(api_url, params=params)

    # Checking if the request was successful (status code 200)
    if response.status_code == 200:
        # Save the image
        image_path = os.path.join(path, f'{image_id}.jpg')
        with open(image_path, 'wb') as f:
            f.write(response.content)
        # print("Image saved successfully at:", image_path)
    else:
        print("Failed to retrieve the image. Status code:", response.status_code)

#operator index
for index, row in df.iterrows():
# for index, row in df.head(100).iterrows():

# resume_index=2870
# for index, row in df.iloc[resume_index:].iterrows():

    ra = row['ra']
    dec = row['dec']
    phot_id = row['specobjid']
    path='D:/galactic_images_production/'
    
    # print(ra, dec, phot_id, path)
    save_image_from_api(ra, dec, 512, 512,'',path, phot_id)
    if index%100==0:
        print(index)
    
