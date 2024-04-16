# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 21:56:52 2024

@author: Matt
"""
import os
import cv2
import numpy as np


folder_path= r'D:\galactic_images_raw'

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        file_path = os.path.join(folder_path, filename)
        o_img=cv2.imread(file_path)
        
        img=np.mean(o_img.copy(),2)
        img=cv2.GaussianBlur(img,(3,3),0)
        img=img.astype(np.uint8)
        
        
        t_value=26
        _, bin_img=cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)

        # Define the seed point for flood fill (center of the galaxy)
        seed_point = (img.shape[1] // 2, img.shape[0] // 2)
        
        # Create a mask for flood fill
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        
        # Flood fill operation
        cv2.floodFill(bin_img, mask, seed_point, 255, 20, 5)
        
        # Invert the mask to get the filled region
        filled_region = cv2.bitwise_not(mask)[1:-1, 1:-1]
        
        # Apply the filled region as a mask on the original image
        central_galaxy = cv2.bitwise_and(bin_img, bin_img, mask=filled_region)
        
        # dot_image=img
        # dot_radius = 5  # Adjust the size of the dot as needed
        # dot_color = (0, 0, 255)  # BGR color format (red)
        
        # cv2.circle(dot_image, (256, 256), dot_radius, dot_color, -1)
        # # Display the result
        
        # cv2.imshow('Dot', dot_image)
        cv2.imshow('Binary', bin_img)
        cv2.imshow('Original', o_img)
        
        cv2.imshow('Central Galaxy', central_galaxy)
        cv2.imshow('filled_region', filled_region)
        cv2.imshow('img', img)
        cv2.imshow('Mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()