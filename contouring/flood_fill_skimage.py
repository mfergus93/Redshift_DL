# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:12:32 2024

@author: m27248
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import os
import cv2

# folder_path= r'D:\galactic_images_raw'
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)
image=cv2.imread('flood_fill_test.png', cv2.IMREAD_GRAYSCALE)

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        file_path = os.path.join(folder_path, filename)
        o_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # Fill a square near the middle with value 127, starting at index (76, 76)
        x=o_img.shape[1]//2
        y=o_img.shape[0]//2
        
        filled = flood_fill(image=o_img, seed_point=(x,y), new_value=255, tolerance=46)
        
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        
        ax[0].imshow(o_img, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        
        ax[1].imshow(filled, cmap=plt.cm.gray)
        ax[1].plot(x, y, 'wo')  # seed point
        ax[1].set_title('After flood fill')
        
        plt.show()