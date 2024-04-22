# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:51:17 2024

@author: m27248
"""

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
        # Construct some test data
        x, y = np.ogrid[-np.pi : np.pi : 100j, -np.pi : np.pi : 100j]
        r = np.sin(np.exp(np.sin(x) ** 3 + np.cos(y) ** 2))
        
        # Find contours at a constant value of 0.8
        contours = measure.find_contours(o_img, 0.8)
        
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(o_img, cmap=plt.cm.gray)
        
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()