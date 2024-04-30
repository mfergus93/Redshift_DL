#color_features
import cv2
import numpy as np

def extract_histogram(image, mask):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram of the masked region
    hist = cv2.calcHist([hsv_image], [0, 1], mask, [180, 256], [0, 180, 0, 256])
    
    # Normalize the histogram
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist.flatten()

def extract_color_histograms(color_image, white_coordinates):
    # Convert the color image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    histograms = []
    for coords in white_coordinates:
        x, y = coords
        # Define a small region around the coordinates
        region_size = 5
        region = hsv_image[y - region_size:y + region_size, x - region_size:x + region_size]
        
        # Create a mask for the region
        mask = np.zeros_like(region[:, :, 0])
        mask[:] = 255
        
        # Extract the color histogram using the mask
        histogram = extract_histogram(region, mask)
        histograms.append(histogram)
    
    return histograms

def extract_color_histogram(color_image, mask_image, threshold):
    # Convert the color image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # Find the indices of white pixels in the mask
    white_indices = np.where(mask_image == threshold)
    
    # Extract the color values from the original image using the mask
    colors = hsv_image[white_indices[0], white_indices[1]]
    
    # Calculate the histogram of the extracted colors
    hist = cv2.calcHist([colors], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize the histogram
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    return hist.flatten()