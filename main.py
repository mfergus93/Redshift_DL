#main routine
import os
import cv2
import numpy as np
import pandas as pd

from Flood_Fill import flood_fill
from Outlining import pavlidis, fillarea
# from y_network import y_network
from color_Features import extract_color_histogram


# labels = pd.read_csv(r'C:\Users\Matt\Desktop\dev\galaxy.csv')

# folder_path= r'D:\galactic_images_raw'
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

image_dir=r'D:\galactic_images_raw'

def preprocess_directory(path):
    
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):# or filename.endswith('.png'):
            
            file_path = os.path.join(path, filename)
            img=cv2.imread(file_path)
            gray_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            flood_img=flood_fill(gray_img)
            # white_coordinates = np.column_stack(np.where(flood_img == 125))
            # contour_img=pavlidis(flood_img)
            print(img.shape)
            print(flood_img.shape)
            histogram = extract_color_histogram(img, flood_img, 125)
            
            return img, flood_img, histogram
            
            
img, flood_img, histogram = preprocess_directory(folder_path)

cv2.imshow('img', img)
cv2.imshow('region', flood_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

