#color_features
import cv2
import numpy as np

def extract_color_histogram(color_image, mask):
    # Convert the color image to HSV color space
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
      
    # Extract the color values from the original image using the mask
    colors = cv2.split(hsv_image)
    
    # Calculate the histogram of the extracted colors
    histSize=65
    h_hist = cv2.calcHist(colors, [0], mask, [histSize], (0, 256))
    s_hist = cv2.calcHist(colors, [1], mask, [histSize], (0, 256))
    v_hist = cv2.calcHist(colors, [2], mask, [histSize], (0, 256))
    
    # Normalize the histogram
    hist_w = 512
    hist_h = 400
    bin_w = int(round( hist_w/histSize ))
    
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    
    # image_size=color_image.shape[0]*color_image.shape[1]
    image_size=np.sum(mask)/255.0

    
    # cv2.normalize(h_hist, h_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(s_hist, s_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    # cv2.normalize(v_hist, v_hist, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    
    for i in range(1, histSize):
        
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(h_hist[i-1]) ),
                            ( bin_w*(i), hist_h - int(h_hist[i]) ),
                            ( 255, 255, 0), thickness=2)
        
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(s_hist[i-1]) ),
                            ( bin_w*(i), hist_h - int(s_hist[i]) ),
                            ( 0, 255, 0), thickness=2)
        
        cv2.line(histImage, ( bin_w*(i-1), hist_h - int(v_hist[i-1]) ),
                            ( bin_w*(i), hist_h - int(v_hist[i]) ),
                            ( 0, 0, 255), thickness=2)
        
    h_hist=np.divide(h_hist, image_size)
    s_hist=np.divide(s_hist, image_size)
    v_hist=np.divide(v_hist, image_size)
    
    return histImage, h_hist, s_hist, v_hist