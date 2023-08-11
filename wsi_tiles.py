# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:24:41 2023

@author: madwi
"""

from openslide import open_slide
import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import time
import psutil
#import cpu_percent

# Start the timer
start_time = time.time()

# Start measuring memory usage
memory_before = psutil.virtual_memory().used
cpu_percent = psutil.cpu_percent()
#Load the slide file (TIFF) into an object.
slide=open_slide("C:/Users/madwi/spyder/Image/3843_Gamma168_S1T0R0_M.tiff")
slide_props=slide.properties
print("these are the slide properties")

#print("vendor is:",slide_props[''])
print(slide_props)
print("Vendor is:", slide_props['openslide.vendor'])


print("Pixel size of X in um is:", slide_props['openslide.mpp-x'])
print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])


#Objective used to capture the image
objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
#magnification level or the lens power used to capture an image in microscopy
print("The objective power is: ", objective)
 


# get slide dimensions for the level 0 i.e. max resolution level
slide_dims = slide.dimensions
print("slide dimensions are ",slide_dims)

#Get a thumbnail of the image and visualize
slide_thumb_600=slide.get_thumbnail(size=(600,600))
#slide_thumb_600.show()
 

#Convert thumbnail to numpy array
slide_thumb_600_np = np.array(slide_thumb_600)
plt.figure(figsize=(8,8))
plt.imshow(slide_thumb_600_np)    



"""
Wed Jul  6 10:01:31 2023

@author: madwi
"""
#Get slide dims at each level. Remember that whole slide images store information
#as pyramid at various levels
dims = slide.level_dimensions

num_levels = len(dims)
print("Number of levels in this image are:", num_levels)

print("Dimensions of various levels in this image are:", dims)

#By how much are levels downsampled from the original image?
factors = slide.level_downsamples
print("Each level is downsampled by an amount of: ", factors)

scaling_factor = 16
best_level = slide.get_best_level_for_downsample(scaling_factor)
best_level_dim = dims[best_level]
print('Number of levels in this image are:', best_level)
print('Dimension of various levels in this image are:', best_level_dim)
best_level_image = slide.read_region((0,0), best_level, best_level_dim)

print('---------------------------------------------------------------------')
#Here it returns the best level to be 2 (third level)
#If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)

#Convert the image to RGB
best_level_image_RGB = best_level_image.convert('RGB')
best_level_image_RGB_np = np.array(best_level_image_RGB)
plt.title('Best Level Image')
plt.imshow(best_level_image_RGB_np)
plt.imsave("Best_Level({})_Image.png".format(best_level), best_level_image_RGB_np)
print('Image saved')

print('---------------------------------------------------------------------')
#######################################################################################################

# #Generating tiles for deep learning training or other processing purposes
# print("############################################################################")

from openslide.deepzoom import DeepZoomGenerator

# #Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=512, overlap=0, limit_bounds=False)
# #Here, we have divided our svs into tiles of size 256 with no overlap.

# #The tiles object also contains data at many levels.
# #To check the number of levels
# print("The number of levels in the tiles object are:", tiles.level_count)


# print("The dimension of the data in each levels are:", tiles.level_dimensions)

# #Total number of tiles in the tiles object
# print("Total number of tiles:", tiles.tile_count)

# #Tiles at particular level
# level_num = 13
print("Tiles shape at level ", best_level, " is: ", tiles.level_tiles[best_level])
print("This means there are ", tiles.level_tiles[best_level][0]*tiles.level_tiles[best_level][1], " total tiles in this level")

# #Dimensions of the tile (tile size) for a specific tile from a specific layer
# tile_dims = tiles.get_tile_dimensions(13, (2,8))
# print(tile_dims)

# #Tile count at the highest resolution level (level 16 in our tiles)
# tile_count_in_large_image = tiles.level_tiles[18]

# #Check tile size for some random tile
# tile_dims = tiles.get_tile_dimensions(18, (200,340))
# print("Dimension of the random tile:", tile_dims)
# #Last tiles may not have full 256x256 dimensions as our large image is not exactly divisible by 256
# tile_dims = tiles.get_tile_dimensions(18, (207,599))
# print("Dimension of the last tile:", tile_dims)


# single_tile = tiles.get_tile(18, (50, 270)) #Provide deep zoom level and address (column, row)
# single_tile_RGB = single_tile.convert('RGB')
# single_tile_RGB.show()


###### Saving each tile to local directory#######################

cols, rows = tiles.level_tiles[18]

print("Columns:", cols, "and Rows:", rows)

import os

tile_dir = "C:/Users/madwi/spyder/new tiles 2/"

for row in range(rows):
    for col in range(cols):
        tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        print("Now saving tile with title: ", tile_name)
        temp_tile = tiles.get_tile(18, (col, 
                                        row))
        
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_np = np.array(temp_tile_RGB)
        plt.imsave(tile_name + ".png", temp_tile_np)


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
print("CPU Usage: {:.2f}%".format(cpu_percent))
#print("CPU usage:", cpu_after - cpu_before, "%")    
