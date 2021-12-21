# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import skimage
from skimage import filters
from scipy import ndimage

# Links
# - Choosing colormaps
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://flothesof.github.io/removing-background-scikit-image.html#Applying-the-watershed
# https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_face_denoise.html
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html


# -- initialize --
path = '3d/10'   # change path to the correct directory of the to be analysed aoa and test setup
file_link = []  # Storage container file links
data_sets = []  # Storage container data sets
pixel_size = 4


# -- Functions --
#  Generates the dataset given a filenumber
def GenArray(file_number):
    data_set = np.genfromtxt(file_link[file_number-1], delimiter=";")
    return data_set[:, :-1]


#  Plots the data set
def ImagePlotter(dataset, line=0):
    plt.imshow(dataset, vmin=np.min(dataset), vmax=np.max(dataset), aspect='auto', cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label('Degrees C')
    plt.title('Differential Infrared Thermography')
    plt.axvline(line, 0, 120, linewidth=2, c="b")
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
averaged_sort = np.mean(np.sort(final)[10:55, 10:100])

final_alt = np.copy(final)
final[final < averaged_sort + 0.15] = averaged_sort + 0.15


# ImagePlotter(averaged_data)
# ImagePlotter(final)


path3d = "3d"
path2d = "2d"
file_link_3d = glob.glob("3d/*")
file_link_2d = glob.glob("2d/*")

# seperate background from foreground and magnify border to obtain chord locations
def FindChord(image):
    edge_detection = filters.sobel(image)
    magnified_edge = filters.gaussian(edge_detection, sigma = 1)
    delta_edge_detection = np.diff(edge_detection)
    delta_magnified_edge_detection = np.diff(magnified_edge)
    jump_locations = np.argwhere(delta_magnified_edge_detection > 0.05)[140:180, 1]
    chord_start, chord_end = np.min(jump_locations), np.max(jump_locations)
    coords_chord = (chord_start, chord_end)
    chord_length = chord_end - chord_start
    return [coords_chord, chord_length]


# Output the image array with magnified/ detailed borders
def FindBorderImage(image, type):
    edge_detection = filters.sobel(image)
    magnified_edge = filters.gaussian(edge_detection, sigma=1)
    if type == 0:
        return edge_detection
    elif type == 1:
        return magnified_edge

# ImagePlotter(final)
# ImagePlotter(final_alt)
# ImagePlotter(averaged_data)
# ImagePlotter(final_alt)
# ImagePlotter(FindBorderImage(final,0))
# ImagePlotter(FindBorderImage(final,0))
# ImagePlotter(FindBorderImage(final,1))

print(FindChord((final)))

def FindTransition(image, start, end, length):
    # Further remove noise to make transition line more visible
    # Using scipiy.ndimage.median_filter
    # and by calling the FindBorderImage function to magnify the edges
    image = ndimage.median_filter(image, 4)
    image = FindBorderImage(image, 1)
    # Correction for ignored distance
    start += 10
    end -= 10
    # crop to only look for a transition edge on the wing surface
    transition_possible = image[10:100,start:end]
    transition_possible[:, 0] += 10
    # Detect transition point based on data jumps and then correct for ignored distance
    transition_detection = np.diff(transition_possible)
    transition_points = np.argwhere(transition_detection > 0.01)
    transition_points[:, 0] += 10
    transition_points[:, 1] += start

    # If transition point exists calculate the location (chordwise and refrence frame)
    if transition_points.tolist():
        tpoint = int(np.mean(transition_points[:, 1]))
        fromleading = end - tpoint + 10
        xoverc1 = fromleading / length
        print(f"end location {end + 10}")
        print(f"loc of transition {tpoint}")
        print(f"from leading edge {fromleading}")
        print(f"as chord ratio {xoverc1}")
        return [True, tpoint]
    else:
        print("No transition")
        return [False]


p = FindTransition(final, FindChord(final)[0][0], FindChord(final)[0][1], FindChord(final)[1])
if p[0]:
    ImagePlotter(final_alt, p[1])
print(p)
print("----------")

# med_denoised = ndimage.median_filter(final, 4)
# med_denoised = FindBorderImage(med_denoised, 1)
# ImagePlotter(med_denoised)
#
# start_location = FindChord(final)[0][0] + 10
# end_location = FindChord(final)[0][1] - 10
# chordL = FindChord(final)[1]
# transition_possible = med_denoised[10:100, start_location:end_location]
# transition_possible[:, 0] += 10
# transition_detection = np.diff(transition_possible)
# transition = np.argwhere(transition_detection > 0.01)
# transition[:, 0] += 10
# transition[:, 1] += start_location
#
# if transition.tolist():
#     transition_point = int(np.mean(transition[:,1]))
#
#     fleading = end_location - transition_point + 10
#     print(f"end location {end_location+10}")
#     xoverc = fleading/chordL
#     print(f"loc of transition {transition_point}")
#     print()
#     print(f"from leading edge {fleading}")
#     print()
#     print(xoverc)
#
#     # final[:, transition_point] = 15.6
#
#     ImagePlotter(final_alt, transition_point)
# else:
#     print("No transition")

# End of script
