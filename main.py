# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import skimage
from skimage import filters

# Links
# - Choosing colormaps
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
#


# -- initialize --
path = '2d/10'   # change path to the correct directory of the to be analysed aoa and test setup
file_link = []  # Storage container file links
data_sets = []  # Storage container data sets
pixel_size = 4


# -- Functions --
#  Generates the dataset given a filenumber
def GenArray(file_number):
    data_set = np.genfromtxt(file_link[file_number-1], delimiter=";")
    return data_set[:, :-1]


#  Plots the data set
def ImagePlotter(dataset):
    plt.imshow(dataset, vmin=np.min(dataset), vmax=np.max(dataset), aspect='auto', cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label('Degrees C')
    plt.title('Differential Infrared Thermography')
    plt.show()


#  Averages all the datasets into one data set
def AverageArrays(datasets):
    return np.mean(datasets, axis=0)


# -- Main code --
#  Generate a file link list
for filename in glob.glob(os.path.join(path, '*.csv')):
    file_link.append(filename)

for data_set in range(len(file_link)):
    data_sets.append(GenArray(data_set))

averaged_data = AverageArrays(data_sets)

final = np.zeros((480//pixel_size, 640//pixel_size))
#final = np.zeros((480, 640))

for x in range(0, len(averaged_data), pixel_size):
    for y in range(0, len(averaged_data[0]), pixel_size):
        sub_array = averaged_data[x:(x + pixel_size), y:(y + pixel_size)]
        sub_array_value = np.mean(sub_array)
        final[int(x/pixel_size), int(y/pixel_size)] = sub_array_value
        #final[int(x):int(int(x) + pixel_size), int(y):int(pixel_size + int(y))] = sub_array_value
averaged_sort = np.mean(np.sort(final)[10:65, 10:100])

final[final < averaged_sort + 0.5] = averaged_sort + 0.25

# ImagePlotter(averaged_data)
# ImagePlotter(final)


path3d = "3d"
path2d = "2d"
file_link_3d = glob.glob("3d/*")
file_link_2d = glob.glob("2d/*")


delta_data = np.diff(final)[10:100]
location = np.argwhere(np.abs(delta_data) > 0.23)
location[:, 0] += 10

y_coord = 10

# if location[y_coord][0] == location[y_coord+1][0]:
#     if abs(location[y_coord][1]-location[y_coord+1][1]) <= 5:
#         np.delete(location[y_coord+1])


# print(location)


# for k in range(len(delta_data)):
#     for l in range(len(delta_data[1])):
#         if abs(delta_data[k])> 0.2:
#             print(delta_data.index()

# seperate background from foreground and magnify border
edge_detection = filters.sobel(final)
magnified_edge = filters.gaussian(edge_detection, sigma = 0.75)


delta_edge_detection = np.diff(edge_detection)
delta_magnified_edge_detection = np.diff(magnified_edge)

location_edge = np.argwhere(np.abs(delta_data) > 0.1)
location_edge[:, 0] += 10

ImagePlotter(edge_detection)

jump_locations = np.argwhere(delta_magnified_edge_detection > 0.1)[140:180,1]
print(jump_locations)
chord_start, chord_end = np.min(jump_locations), np.max(jump_locations)
coords_chord = (chord_start, chord_end)
chord_length = chord_end - chord_start
print(coords_chord)
print(chord_length)
ImagePlotter(magnified_edge)