# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:40:44 2024

@author: Matt
"""
from sdss import Region

ra = 179.689293428354
dec = -0.454379056007667

reg = Region(ra, dec, fov=0.0033)

reg.show()
reg.show3b()