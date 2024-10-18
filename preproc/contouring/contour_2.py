import os
import cv2
import numpy as np

def resize_image(image_path, new_size):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Gaussian blur to smooth the image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Thresholding to get the gray area
    _, bin_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    
    # Finding contours of the gray area
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    return contour_img

# Example usage:
image_path = "test_galaxy4.jpg"  # Path to your input image
new_size = (100, 100)  # New size for the resized image
resized_image = resize_image(image_path, new_size)

if resized_image is not None:
    # Display or save the resized image
    cv2.imshow("Resized Image", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No gray area found in the image.")