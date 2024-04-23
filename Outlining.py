#traces outline of central galaxy/object using theo pavlidis on a binarized image!
import cv2
import numpy as np
import os

# Read the image
img = cv2.imread(r'D:\galactic_images\299491326712375296.jpg',cv2.IMREAD_GRAYSCALE)

def pavlidis(img): #pass me a binarized image and I will trace the outline
    
    c=0 #trace indicator, 0 means proceed with tracing, 1 means stop
    x=img.shape[1]//2 #Initialize the x value to the center of the image, coincides with the galaxy of interest
    y=img.shape[0]//2 #Initialize the y value to the center of the image
    bimg=np.zeros(img.shape[:2])#create an empty image for us to draw the outline on

    # while c == 0: #we want to exit the center and find the perimeter, this loop will extend us right to the perimeter
    #     x+=1#Assumed the center is 255, take us right
    #     if img[y,x]!=255:#Check if we've reached the edge of the whitespace and located the first black pixel going right
    #         c=1
    #         x=x-1
    
    
    while c == 0: #we want to exit the center and find the perimeter, this loop will extend us right to the perimeter

        if img[y,x]==255:
            x=x-1
        if img[y,x]!=255:#Check if we've reached the edge of the whitespace and located the first black pixel going right
            x=x+1
            c=1
    # while c ==0:
    #     if img[y,x]!=0:
    #         y=y-1
    #     else:
    #         c=1
    #Pavlidis requires that the pixel to the left of our initial starting positions is non-white
    
    print(x,y)
    
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
            direction=directions[(((directions.index(direction)-1))%4)] #change direction CCW
        elif b1_value==0 and b2_value==0 and b3_value==0:
            direction=directions[(((directions.index(direction)+1))%4)] #change direction CW
        elif b2_value==255:
            bimg[y,x]=255
        elif b3_value==255:
            bimg[y,x]=255
        
        print(direction, x, y)
        for i, value in enumerate(reversed(block_values)):
            if value==255:
                input = block_coords[2-i]
        print(input)
        result=np.append(result,[input],axis=0)

    result, ind=np.unique(result,axis=0, return_index=True)
    result=result[np.argsort(ind)]
    return (result)

def fillarea(ctr):
    maxx = np.max(ctr[:, 0]) + 1
    maxy = np.max(ctr[:, 1]) + 1
    contourImage = np.zeros( (maxy, maxx) )
    length = ctr.shape[0]
    for count in range(length):
        contourImage[ctr[count, 1], ctr[count, 0]] = 255
        cv2.line(contourImage, (ctr[count, 0], ctr[count, 1]), \
        (ctr[(count + 1) % length, 0], ctr[(count + 1) % length, 1]), \
        (255, 0, 255), 1)
    fillMask = cv2.copyMakeBorder(contourImage, 1, 1, 1, 1, \
    cv2.BORDER_CONSTANT, 0).astype(np.uint8)
    areaImage = np.zeros((maxy, maxx), np.uint8)
    startPoint = (int(maxy/2), int(maxx/2))
    cv2.floodFill(areaImage, fillMask, startPoint, 128)
    area = np.sum(areaImage)/128
    return area

folder_path= r'D:\galactic_images_raw'

# for filename in os.listdir(folder_path):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
        
filename='1195806900076177408.jpg'
file_path = os.path.join(folder_path, filename)
image=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

t_value=25
_, bin_img=cv2.threshold(image, t_value, 255, cv2.THRESH_BINARY)

cv2.imshow('bin', bin_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

contour=pavlidis(bin_img)

xc=np.float64(contour[:,1])
yc=np.float64(contour[:,0])

blank_img=np.zeros(img.shape[:2])
ctr_img=cv2.polylines(blank_img,[contour], True, 255,1)

cv2.floodFill(ctr_img, None, (256,256), 255)

