import cv2
import os
import numpy as np

def flood_fill_recursive(image, x, y, old_value, new_value, mask, lo_diff, up_diff):
    if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
        return  # Out of bounds

    if mask[y, x] == 1:
        return  # Already filled

    pixel_value = image[y, x]
    if old_value - lo_diff <= pixel_value <= old_value + up_diff:
        image[y, x] = new_value
        mask[y, x] = 1

        flood_fill_recursive(image, x + 1, y, old_value, new_value, mask, lo_diff, up_diff)
        flood_fill_recursive(image, x - 1, y, old_value, new_value, mask, lo_diff, up_diff)
        flood_fill_recursive(image, x, y + 1, old_value, new_value, mask, lo_diff, up_diff)
        flood_fill_recursive(image, x, y - 1, old_value, new_value, mask, lo_diff, up_diff)

def flood_fill(image, seed_point, new_value, lo_diff, up_diff):
    old_value = image[seed_point[1], seed_point[0]]

    mask = np.zeros_like(image)  # Mask to track filled pixels

    flood_fill_recursive(image, seed_point[0], seed_point[1], old_value, new_value, mask, lo_diff, up_diff)

# # Load a grayscale image
# image = cv2.imread('path_to_your_grayscale_image.jpg', cv2.IMREAD_GRAYSCALE)
# folder_path= r'D:\galactic_images_raw'

# for filename in os.listdir(folder_path):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
        
#         file_path = os.path.join(folder_path, filename)
#         o_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
#         # img=np.mean(o_img.copy(),2)
#         # img=cv2.GaussianBlur(img,(3,3),0)
#         image=o_img.astype(np.uint8)
        
        
image=cv2.imread(r'D:\flood_fill_test.png', cv2.IMREAD_GRAYSCALE)

    seed_point = (image.shape[1] // 2, image.shape[0] // 2)
    new_value = 255  # New intensity value for flood fill
    lo_diff = 10
    up_diff = 10

    flood_fill(image, seed_point, new_value, lo_diff, up_diff)

    cv2.imshow("Filled Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()