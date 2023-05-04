"""
    Generic functions to avoid repetition and make the code neat
"""

import matplotlib as mpl


def reverse_colourmap(comap, name='my_cmap_r'):
    # reverse the color scale in the plot
    reverse = []
    k = []

    for key in comap._segmentdata:
        k.append(key)
        channel = comap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r


def listToString(s):
    # transforms array or list of strings to only one string parameter
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1
