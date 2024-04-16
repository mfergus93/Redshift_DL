#traces outline of central galaxy/object using theo pavlidis on a binarized image!
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the image
img = cv2.imread(r'D:\galactic_images\299491326712375296.jpg',cv2.IMREAD_GRAYSCALE)

def pavlidis(img): #pass me a binarized image and I will trace the outline
    
    c=0 #trace indicator, 0 means proceed with tracing, 1 means stop
    x=img.shape[1]//2 #Initialize the x value to the center of the image, coincides with the galaxy of interest
    y=img.shape[0]//2 #Initialize the y value to the center of the image
    bimg=np.zeros(img.shape[:2])#create an empty image for us to draw the outline on

    while c == 0: #we want to exit the center and find the perimeter, this loop will extend us right to the perimeter
        x+=1#Assumed the center is 255, take us right
        if img[y,x]!=255:#Check if we've reached the edge of the whitespace
            c=1 #if so set the flag

    #Now that we are on the edge of the perimeter we can trace the outline
    
    b1_value,b2_value,b3_value = 0,0,0 #Initial values
    b1_coord,b2_coord,b3_coord= 0,0,0 #Initial coordinates
    directions = 'up', 'right', 'down', 'left'#set of moves we can make
    direction='right' #first move
    
    input=(y,x)
    result=np.array([input])

    c=0 #Reinitalize stop indicator for the perimeter portion of the loop
    start_pos=input
    while c<2:

        y,x=input
        if input==list(start_pos):
            c=c+1
            
        #Check the direction forward and to the left/right to make a decision
        if direction =='up':
            b1_value, b1_coord=img[y-1,x-1], [y-1,x-1]
            b2_value, b2_coord=img[y-1,x], [y-1,x]
            b3_value, b3_coord=img[y-1,x+1], [y-1,x+1]
        elif direction =='right':
            b1_value, b1_coord=img[y-1,x+1], [y-1,x+1]
            b2_value, b2_coord=img[y,x+1], [y,x+1]
            b3_value, b3_coord=img[y+1,x+1], [y+1,x+1]
        elif direction =='down':
            b1_value, b1_coord=img[y+1,x+1], [y+1,x+1]
            b2_value, b2_coord=img[y+1,x], [y+1,x]
            b3_value, b3_coord=img[y+1,x-1], [y+1,x-1]
        elif direction =='left':
            b1_value, b1_coord=img[y+1,x-1], [y+1,x-1]
            b2_value, b2_coord=img[y,x-1], [y,x-1]
            b3_value, b3_coord=img[y-1,x-1], [y-1,x-1]

        block_values= b1_value,b2_value,b3_value
        block_coords=b1_coord,b2_coord,b3_coord #pack these up into list

        if b1_value==255:
            bimg[y,x]=255
            direction=directions[(((directions.index(direction)-1))%4)]
        elif b1_value==0 and b2_value==0 and b3_value==0:
            direction=directions[(((directions.index(direction)+1))%4)]
        elif b2_value==255:
            bimg[y,x]=255
        elif b3_value==255:
            bimg[y,x]=255

        for i, value in enumerate(reversed(block_values)):
            if value==255:
                input = block_coords[2-i]

        result=np.append(result,[input],axis=0)

    result, ind=np.unique(result,axis=0, return_index=True)
    result=result[np.argsort(ind)]
    return (result)


t_value=25
_, bin_img=cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)
cv2.imshow('binarfy', bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
result=pavlidis(bin_img)