#
import os
import cv2
import numpy as np

# folder_path= r'D:\galactic_images_raw'
# folder_path=os.path.dirname(os.path.realpath(__file__))
# os.chdir(folder_path)

def flood_fill(o_img):

        img=cv2.GaussianBlur(o_img.copy(),(3,3),0)
        img=img.astype(np.uint8)
        
        #create a mask where we will never consider values under 6 or is it 6 and under?
        t_value=30
        _, bin_img=cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)
        # mask = cv2.bitwise_not(bin_img)
        mask = np.zeros_like(bin_img)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

        # Define the seed point for flood fill (center of the galaxy)
        seed_point = (img.shape[1] // 2, img.shape[0] // 2)
        
        # Flood fill operation
        cv2.floodFill(image=img, mask=mask, seedPoint=seed_point, newVal=255, \
                      loDiff=50, upDiff=10, flags= 8 | ( 255 << 8 ) | cv2.FLOODFILL_FIXED_RANGE)
        
        mask=mask[1:-1, 1:-1]
        
        return mask
        
# # folder_path= r'D:\galactic_images_raw'
# folder_path=os.path.dirname(os.path.realpath(__file__))
# os.chdir(folder_path)

# for filename in os.listdir(folder_path):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
        
#         file_path = os.path.join(folder_path, filename)
#         o_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        

#         img=cv2.GaussianBlur(o_img.copy(),(3,3),0)
#         img=img.astype(np.uint8)
        
#         #create a mask where we will never consider values under 6 or is it 6 and under?
#         t_value=60
#         _, bin_img=cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)
#         mask = cv2.bitwise_not(bin_img)
#         mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

#         # Define the seed point for flood fill (center of the galaxy)
#         seed_point = (img.shape[1] // 2, img.shape[0] // 2)
        
#         # Flood fill operation
#         cv2.floodFill(image=img, mask=mask, seedPoint=seed_point, newVal=125, \
#                       loDiff=50, upDiff=2, flags= 8 | ( 125 << 8 ) | cv2.FLOODFILL_FIXED_RANGE)
        
#         # Invert the mask to get the filled region
#         # filled_region = cv2.bitwise_not(mask)[1:-1, 1:-1]
#         # Apply the filled region as a mask on the original image
#         # central_galaxy = cv2.bitwise_and(bin_img, bin_img, mask=filled_region)
        
#         # Display the result        
#         cv2.imshow('Binary', bin_img)
#         cv2.imshow('Original', o_img)
        
#         # cv2.imshow('Central Galaxy', central_galaxy)
#         # cv2.imshow('filled_region', filled_region)
#         cv2.imshow('img', img)
#         cv2.imshow('Mask', mask)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# # Example usage:
# image_path = "test_galaxy4.jpg"  # Path to your input image
# o_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in color mode

# filled_region = flood_fill(o_img)

# # Display the filled region
# cv2.imshow("Filled Region", filled_region)
# cv2.waitKey(0)
# cv2.destroyAllWindows()