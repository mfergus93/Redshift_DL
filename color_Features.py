#color_features
import cv2
import numpy as np

def extract_color_histogram(color_image, mask_image, threshold):
    # Convert the color image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
      
    # Extract the color values from the original image using the mask
    colors = cv2.split(hsv_image)
    
    # Calculate the histogram of the extracted colors
    cv2.imshow('img', color_image)
    cv2.imshow('region', mask_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    histSize = 256
    h_hist = cv2.calcHist(colors, [0], mask_image, [256], (0, 256))
    s_hist = cv2.calcHist(colors, [1], mask_image, [256], (0, 256))
    v_hist = cv2.calcHist(colors, [2], mask_image, [256], (0, 256))

    
    # Normalize the histogram
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    
    cv2.normalize(h_hist, h_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(v_hist, v_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    
    for i in range(1, histSize):
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(h_hist[i-1]) ),
        ( bin_w*(i), hist_h - int(h_hist[i]) ),
        ( 255, 0, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(s_hist[i-1]) ),
        ( bin_w*(i), hist_h - int(s_hist[i]) ),
        ( 0, 255, 0), thickness=2)
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(v_hist[i-1]) ),
        ( bin_w*(i), hist_h - int(v_hist[i]) ),
        ( 0, 0, 255), thickness=2)
    cv2.imshow('Source image', color_image)
    cv2.imshow('calcHist Demo', histImage)
    cv2.waitKey()

    
    return histImage