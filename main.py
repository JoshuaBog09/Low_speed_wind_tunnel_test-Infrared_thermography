# Imports
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from skimage import filters
from scipy import ndimage

# Links
# - Choosing colormaps
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# - Image processing
# https://flothesof.github.io/removing-background-scikit-image.html#Applying-the-watershed
# https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_face_denoise.html
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html


# -- Functions --
#  Generates the dataset given a filenumber
def GenArray(file_number):
    data_set = np.genfromtxt(file_link[file_number-1], delimiter=";")
    return data_set[:, :-1]


#  Plots the data set
def ImagePlotter(dataset, line=0, title = ""):
    plt.imshow(dataset, vmin=np.min(dataset), vmax=np.max(dataset), aspect='auto', cmap='inferno')
    cbar = plt.colorbar()
    cbar.set_label('Degrees C')
    plt.title(f"Differential Infrared Thermography at {title} degrees")
    plt.axvline(line, 0, 120, linewidth=2, c="b")
    plt.show()


#  Averages all the datasets into one data set
def AverageArrays(datasets):
    return np.mean(datasets, axis=0)


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

def FindTransition(image, start, end, length, alter = 0, modealter = "3d"):
    # Further remove noise to make transition line more visible
    # Using scipiy.ndimage.median_filter
    # and by calling the FindBorderImage function to magnify the edges
    image = ndimage.median_filter(image, 6)
    #ImagePlotter(image)
    image = FindBorderImage(image, 1)
    #ImagePlotter(image)
    # Correction for ignored distance
    start += 10
    end -= 5
    # crop to only look for a transition edge on the wing surface
    transition_possible = image[10:100,start:end]
    transition_possible[:, 0] += 10
    # Detect transition point based on data jumps and then correct for ignored distance
    transition_detection = np.diff(transition_possible)
    if modealter.strip() == "3d":
        if alter == 0:
            transition_points = np.argwhere((transition_detection > 0.0122) & (transition_detection < 0.0175))
        elif alter == 1:
            transition_points = np.argwhere((transition_detection > 0.00815) & (transition_detection < 0.0175))
    elif modealter.strip() == "2d":
        if alter == 0:
            transition_points = np.argwhere((transition_detection > 0.012) & (transition_detection < 0.0175))
        elif alter == 1:
            transition_points = np.argwhere((transition_detection > 0.00815) & (transition_detection < 0.015))

    transition_points[:, 0] += 10
    transition_points[:, 1] += start

    # If transition point exists calculate the location (chordwise and refrence frame)
    if transition_points.tolist():
        tpoint = int(np.mean(transition_points[:, 1]))
        fromleading = end - tpoint + 10
        tpointaschord = fromleading / length
        print(f"end location {end + 10}")
        print(f"loc of transition {tpoint}")
        print(f"from leading edge {fromleading}")
        print(f"as chord ratio {tpointaschord}")
        return [True, tpoint, tpointaschord]
    else:
        print("Transition at leading edge")
        tpoint = end + 5
        tpointaschord = 0
        return [False, tpoint, tpointaschord]


def PlotTransitionVariation(x_list, y_list, mode):
    plt.plot(x_list, y_list)
    plt.grid()
    plt.title(f"Chord wise variation of transition point for angles of attack for {mode} set up")
    plt.xlabel("x/c [-]")
    plt.ylabel("Angle of attack [degrees]")
    plt.show()

# ---- Main code ----

# -- initialize values --
path = '3d/0'   # change path to the correct directory of the to be analysed aoa and test setup
file_link = []  # Storage container file links
data_sets = []  # Storage container data sets
xovercpoints = [] # Storage container x over c transition location
pixel_size = 4

path3d = "3d"
path2d = "2d"
file_link_3d = glob.glob("3d/*")
file_link_2d = glob.glob("2d/*")
title_names = []  # for graph title

# Generate names for the graph title
for name in file_link_2d:
    title_names.append(name[3:])

for path, title_name in zip(file_link_2d, title_names):
    file_link = []
    data_sets = []

    #  Generate a file link list
    for filename in glob.glob(os.path.join(path, '*.csv')):
        file_link.append(filename)

    # print(file_link)
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

    print(FindChord(final), title_name)

    if mode.strip() == "3d":
        if title_name.strip() == "7" or title_name.strip() == "5" or title_name.strip() == "5.5" or title_name.strip() == "6.5":
            alter_factor = 1
        else:
            alter_factor = 0

    if mode.strip() == "2d":
        if title_name.strip() == "5" or title_name.strip() == "5.5" or title_name.strip() == "6" or title_name.strip() == "6.5":
            alter_factor = 1
        else:
            alter_factor = 0

    p = FindTransition(final, FindChord(final)[0][0], FindChord(final)[0][1], FindChord(final)[1], alter_factor, mode)
    if p[0]:
        ImagePlotter(final_alt, p[1], title_name)
    elif not p[0]:
        ImagePlotter(final_alt, p[1], title_name)
    xovercpoints.append(p[2])
    print("----------")

b_or_not = []
title_names2 = []

for part in title_names:
    if part[-1] == "b":
        title_names2.append(part.strip("b"))
        b_or_not.append(1)
    else:
        b_or_not.append(0)
        title_names2.append(part)

title_names_angle = list(map(float, title_names2))

for i, bfound in zip(range(len(title_names_angle)), b_or_not):
    if bfound == 1:
        title_names_angle[i] *= 10

bubble = True

while bubble:
    bubble = False
    for i in range(len(title_names_angle)-1):
        if title_names_angle[i+1] < title_names_angle[i]:
            temp_file = title_names_angle[i+1]
            title_names_angle[i + 1] = title_names_angle[i]
            title_names_angle[i] = temp_file

            temp_file_1 = b_or_not[i + 1]
            b_or_not[i + 1] = b_or_not[i]
            b_or_not[i] = temp_file_1

            temp_file_2 = xovercpoints[i+1]
            xovercpoints[i + 1] = xovercpoints[i]
            xovercpoints[i] = temp_file_2

            bubble = True

b_values = title_names_angle[b_or_not.index(1):]
xoverc_values = xovercpoints[b_or_not.index(1):]

del title_names_angle[b_or_not.index(1):]
del xovercpoints[b_or_not.index(1):]

for i in range(len(b_values)):
    b_values[i] /= 10

bubble2 = True
while bubble2:
    bubble2 = False
    for i in range(len(b_values) - 1):
        if b_values[i+1] > b_values[i]:
            temp_file_3 = b_values[i+1]
            b_values[i + 1] = b_values[i]
            b_values[i] = temp_file_3

            temp_file_4 = xoverc_values[i + 1]
            xoverc_values[i + 1] = xoverc_values[i]
            xoverc_values[i] = temp_file_4

            bubble2 = True

title_names_angle.extend(b_values)
xovercpoints.extend(xoverc_values)

PlotTransitionVariation(xovercpoints, title_names_angle, mode)

# End of script
