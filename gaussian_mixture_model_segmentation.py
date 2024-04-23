# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:45:25 2024

@author: m27248
"""

##comment!

import os
import cv2
import numpy as np


# folder_path= r'D:\galactic_images_raw'
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        file_path = os.path.join(folder_path, filename)
        o_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        

        img=cv2.GaussianBlur(o_img.copy(),(3,3),0)
        img=img.astype(np.uint8)
        
        #create a mask where we will never consider values under 6 or is it 6 and under?
        t_value=60
        _, bin_img=cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(bin_img)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)