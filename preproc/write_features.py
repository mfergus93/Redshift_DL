#takes a directory of imgcutouts and returns their histograms in rgb/hsv
import os
import cv2
import numpy as np
from flood_fill import flood_fill
from color_features import extract_rgb_histogram, extract_hsv_histogram
import pickle
import pandas as pd

def preprocess_directory(path, color, flood_fill_flag, size): #probably change this to preprocess image or batch of images
    
    output_list=[]
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):# or filename.endswith('.png'):
            
            file_path = os.path.join(path, filename)
            img=cv2.imread(file_path)
            gray_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            img=crop_image(img, size)
            gray_img=crop_image(gray_img, size)
            
            if flood_fill_flag == True:
                #central flood fill mask
                mask=flood_fill(gray_img)
            elif flood_fill_flag == False:
                #No mask
                mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
                
            # contour_img=pavlidis(flood_img)
            
            if color == 'rgb':
                histogram_image, h_hist, s_hist, v_hist = extract_rgb_histogram(img, mask)
            if color == 'hsv':
                histogram_image, h_hist, s_hist, v_hist = extract_hsv_histogram(img, mask)

            # output_list.append([img, flood_img, histogram_image, h_hist, s_hist, v_hist])
            feature_array=np.concatenate((h_hist, s_hist, v_hist), axis=0)
            output_list.append([filename, feature_array])
            
            # cv2.imshow('Galactic Image', histogram_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
    return output_list


#write all preprocessed images as pickled features for the model to a directory 
def write_features(features, filename):
    # Create the features subfolder if it doesn't exist
    if not os.path.exists('features'):
        os.makedirs('features')
    
    # Define the file path to save the features.pkl file inside the features subfolder
    file_path = os.path.join('features', filename)

    # Write features to the file
    with open(file_path, 'wb') as f:
        pickle.dump(features, f)


#call in the prerocessed features for the model
def load_features(feature_directory, filename):
    
    features_path = os.path.join(feature_directory, filename)
    with open(features_path, 'rb') as input_file:
        features = pickle.load(input_file)
    x = features
    
    labels = pd.read_csv(r'galaxy.csv')
    y=labels.iloc[:, [18, -1]]
    
    return x, y

# def crop_images(images, target_size):
#     cropped_images = []
#     for image in images:
#         height, width, _ = image.shape
#         # Calculate the coordinates for cropping the center of the image
#         start_y = max(0, height // 2 - target_size[0] // 2)
#         end_y = start_y + target_size[0]
#         start_x = max(0, width // 2 - target_size[1] // 2)
#         end_x = start_x + target_size[1]
#         # Crop the image
#         cropped_image = image[start_y:end_y, start_x:end_x]
#         cropped_images.append(cropped_image)
#     return np.array(cropped_images)

def crop_image(image, target_size):
    height = image.shape[0]
    width = image.shape[1]
    
    # Ensure the target size does not exceed the image size
    crop_height = min(target_size, height)
    crop_width = min(target_size, width)
    
    # Calculate the coordinates for cropping the center of the image
    start_y = max(0, height // 2 - crop_height // 2)
    end_y = start_y + crop_height
    start_x = max(0, width // 2 - crop_width // 2)
    end_x = start_x + crop_width
    
    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return np.array(cropped_image)