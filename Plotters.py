# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 11:11:26 2024

@author: M27248
"""

import pandas as pd
import numpy as np

df=pd.read_csv('galaxy.csv')


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame is named df and the column with redshift values is named 'redshift'
sns.histplot(df['redshift'], kde=True)

plt.title('Distribution of Redshift Values')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.show()




import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.coordinates import SkyCoord
import astropy.units as u


ra=np.array(df['ra'])*u.deg
dec=np.array(df['dec'])*u.deg


plt.style.use(astropy_mpl_style)

sky_coords = SkyCoord(ra=ra, dec=dec, frame='icrs')

plt.figure(figsize=(10, 5))
plt.subplot(111, projection="mollweide")
plt.scatter(sky_coords.ra.wrap_at(180*u.degree).radian, sky_coords.dec.radian, s=1, color='blue')
plt.grid(True)
plt.title('Sky Visualization Map')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()