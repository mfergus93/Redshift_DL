# Computer Vision HW4
import cv2
import numpy as np
import os

def niblack_threshold(image, window_size, k):
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the local mean and standard deviation using a rectangular window
    mean = cv2.blur(image.astype(np.float64), (window_size, window_size))
    mean_square = cv2.blur(np.square(image).astype(np.float64), (window_size, window_size))
    stddev = np.sqrt(mean_square - np.square(mean))
    
    # Calculate the Niblack threshold
    threshold = mean + k * stddev
    
    # Binarize the image using the calculated threshold
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 255
    
    return binary_image

def bernsen_threshold(image, window_size, contrast_threshold):
    # Convert the image to grayscale if it's in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Pad the image to handle edge cases
    padded_image = cv2.copyMakeBorder(image, window_size // 2, window_size // 2, window_size // 2, window_size // 2, cv2.BORDER_CONSTANT, value=255)
    
    # Initialize the binary image
    binary_image = np.zeros_like(image)
    
    # Iterate over image pixels
    for i in range(window_size // 2, padded_image.shape[0] - window_size // 2):
        for j in range(window_size // 2, padded_image.shape[1] - window_size // 2):
            # Extract the local window
            window = padded_image[i - window_size // 2:i + window_size // 2 + 1, j - window_size // 2:j + window_size // 2 + 1]
            
            # Calculate the local minimum and maximum intensities
            local_min = np.min(window)
            local_max = np.max(window)
            
            # Calculate the Bernsen threshold
            threshold = (local_min + local_max) // 2
            
            # Determine the binary value based on the contrast threshold
            binary_value = 0 if (local_max - local_min) < contrast_threshold else 255
            
            # Update the binary image
            binary_image[i - window_size // 2, j - window_size // 2] = binary_value
    
    return binary_image

folder_path= r'D:\galactic_images_raw'

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        file_path = os.path.join(folder_path, filename)
        o_img=cv2.imread(file_path)
        img=np.mean(o_img.copy(),2)
        img=cv2.GaussianBlur(img,(3,3),0)
        img=img.astype(np.uint8)
        
        inner_region = o_img[(o_img.shape[0] - 100) // 2 : (o_img.shape[0] + 100) // 2,
                             (o_img.shape[1] - 100) // 2 : (o_img.shape[1] + 100) // 2]
        
        threshold,img=cv2.threshold(img,0,255, cv2.THRESH_OTSU)
        img=cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_CONSTANT,None,value=(0,0,0))
        img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\Binary_'+filename,img)
        
        # bimg=bernsen_threshold(img, 15, 50)
        # img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\Binary_Bernsen_'+filename,bimg)
        
        # nimg=niblack_threshold(img, 15, -0.2)
        # img_out=cv2.imwrite(r'C:\Users\Matt\Desktop\Results\Binary_Niblack_'+filename,nimg)
        
        
        
        bimg=np.zeros(img.shape[:2])