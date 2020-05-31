#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 08:59:16 2020

@author: tirsgaard
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib

save_path = "saved_frames/png_dataset/test"
load_path = "saved_frames"
load_extension = "Riverraid-v0_test"


import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Find files
import glob, os
i = 0
print(save_path + "/" + load_extension + "*.npy")
for file in glob.glob(load_path + "/" + load_extension + "*.npy"):
    print("Converting file: " + file)
    # Import array
    data_files = np.load(file)

    # Convert to png files
    for j in range(data_files.shape[0]):
        matplotlib.image.imsave(save_path + '/pic'+str(i)+ ".png", data_files[j], vmin = 0, vmax = 255)
        i += 1
    #Image.fromarray(data_files[0]).save(save_path + '/pic'+str(i)+ ".png", 'PNG')
    
    
