#
import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import linalg
import itertools
color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9.0, 5.0)
    plt.ylim(-3.0, 6.0)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# folder_path= r'D:\galactic_images_raw'
folder_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        
        file_path = os.path.join(folder_path, filename)
        o_img=cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        

        img=cv2.GaussianBlur(o_img.copy(),(3,3),0)
        img=img.astype(np.uint8)
        
        #create a mask where we will never consider values under 6 or is it 6 and under?
        t_value=60
        _, bin_img=cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_not(bin_img)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        
        seed_point = (img.shape[1] // 2, img.shape[0] // 2)
        
        # Flood fill operation
        cv2.floodFill(image=bin_img, mask=mask, seedPoint=seed_point, newVal=125, \
                      loDiff=255, upDiff=255, flags= 8 | ( 125 << 8 ) | cv2.FLOODFILL_FIXED_RANGE)
        
        #Extract the location of the filled region, which will be gray 125
        xy_coords = np.flip(np.column_stack(np.where(bin_img == 125)), axis=1)
        
        #append the list by the co-ordinate by its intensity.
        ixy=[]        
        for i in xy_coords:
            ixy.append([img[i[1],i[0]],i[0],i[1]])
        
        X=[]
        for i in ixy:
            intensity=i[0]
            x=i[1]
            y=i[2]
            
            for j in range(intensity):
                X.append([x,y])
                
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)

        plot_results(
        X,
        gm.predict(X),
        gm.means_,
        gm.covariances_,
        1,
        "Plot")
        
        # cv2.imshow('Binary', bin_img)
        # cv2.imshow('Original', o_img)
        
        # # cv2.imshow('Central Galaxy', central_galaxy)
        # # cv2.imshow('filled_region', filled_region)
        # cv2.imshow('img', img)
        # cv2.imshow('Mask', mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        #X is the region inside the center binary hole, and it must be reorganized
        #in a way such that there are intensity*(x,y) instances in an array to
        #input into the GM model I think
        
        # gm=GaussianMixture(n_components=2, random_state=0).fit(X)
        # gm.means_