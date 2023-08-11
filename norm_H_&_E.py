# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:18:16 2023

@author: madwi
"""

from openslide import open_slide
import openslide
import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import time
import psutil
#import cpu_percent

# Start the timer
start_time = time.time()

# Start measuring memory usage
memory_before = psutil.virtual_memory().used

# Start measuring CPU usage
#cpu_before = psutil.cpu_percent(interval=None)

# Function to normalize the tile
def normalize_tile(tile):
    # Normalize the tile using OpenCV
    normalized_tile = cv2.normalize(tile, None, alpha=1, beta=0.15, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_tile

# Function to extract H and E signals from the normalized tile
def extract_h_e_signals(normalized_tile):
    # Convert the normalized tile to the LAB color space
    lab_tile = cv2.cvtColor(normalized_tile, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into channels
    l_channel, a_channel, b_channel = cv2.split(lab_tile)
    
    # Extract the H and E signals from the LAB channels
    h_channel = l_channel
    e_channel = cv2.subtract(l_channel, a_channel)
    
    return h_channel, e_channel

# Path to the folder containing the tiles
tiles_folder = 'C:/Users/madwi/spyder/new tiles 2'

# Path to save the normalized tiles, H signals, and E signals
# output_folder1 = 'path/to/output/folder'

# Iterate over the tiles in the folder
for filename in os.listdir(tiles_folder):
    # Load the tile image
    tile_path = os.path.join(tiles_folder, filename)
    tile = cv2.imread(tile_path)
    
    # Normalize the tile
    normalized_tile = normalize_tile(tile)
    
    # Extract H and E signals from the normalized tile
    h_channel, e_channel = extract_h_e_signals(normalized_tile)
    
    # Save the normalized tile, H signal, and E signal
    normalized_tile_path = os.path.join('C:/Users/madwi/spyder/new tiles 2/Normalized', f'normalized_{filename}')
    h_signal_path = os.path.join('C:/Users/madwi/spyder/new tiles 2/H stain', f'h_signal_{filename}')
    e_signal_path = os.path.join('C:/Users/madwi/spyder/new tiles 2/E stain', f'e_signal_{filename}')
    
    cv2.imwrite(normalized_tile_path, normalized_tile)
    cv2.imwrite(h_signal_path, h_channel)
    cv2.imwrite(e_signal_path, e_channel)

    print(f"Processed {filename}")
    
end_time = time.time()

# Stop measuring memory usage
memory_after = psutil.virtual_memory().used

# Stop measuring CPU usage
#cpu_after = psutil.cpu_percent(interval=None)

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Calculate the memory usage
memory_used = memory_after - memory_before

print("Processing completed.")
# Print the results
print("Time taken:", elapsed_time, "seconds")
print("Memory used:", memory_used, "bytes")
#print("CPU usage:", cpu_after - cpu_before, "%")    


