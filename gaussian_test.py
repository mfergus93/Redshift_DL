import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy import linalg
import itertools
import matplotlib.patches as patches

color_iter = itertools.cycle(["navy", "c", "cornflowerblue", "gold", "darkorange"])

def plot_results(X, Y_, means, covariances, index, title):
    plt.figure(figsize=(10, 5))
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = patches.Ellipse(
            mean, v[0], v[1], angle=180.0 + angle, color='none', linestyle='-', linewidth=2
        )
        ell.set_clip_box(splot.bbox)
        splot.add_artist(ell)

    plt.xlim(0, img.shape[1])
    plt.ylim(0, img.shape[0])
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

folder_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(folder_path)

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        file_path = os.path.join(folder_path, filename)
        o_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.GaussianBlur(o_img.copy(), (3, 3), 0)
        
        t_value = 60
        _, bin_img = cv2.threshold(img, t_value, 255, cv2.THRESH_BINARY)
        
        mask = cv2.bitwise_not(bin_img)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        
        seed_point = (img.shape[1] // 2, img.shape[0] // 2)
        cv2.floodFill(image=bin_img, mask=mask, seedPoint=seed_point, newVal=125,
                      loDiff=255, upDiff=255, flags= 8 | ( 125 << 8 ) | cv2.FLOODFILL_FIXED_RANGE)
        
        xy_coords = np.flip(np.column_stack(np.where(bin_img == 125)), axis=1)
        
        ixy = []
        for i in xy_coords:
            ixy.append([img[i[1], i[0]], i[0], i[1]])
        
        X = []
        for i in ixy:
            intensity = i[0]
            x = i[1]
            y = i[2]
            
            for _ in range(intensity):
                X.append([x, y])
        
        X = np.array(X)
        
        gm = GaussianMixture(n_components=2, random_state=0).fit(X)
        
        plot_results(
            X,
            gm.predict(X),
            gm.means_,
            gm.covariances_,
            1,
            "Gaussian Mixture Model"
        )
        
        plt.imshow(o_img, cmap='gray')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig("output.png", bbox_inches = 'tight', pad_inches = 0, dpi=300)
        plt.close()