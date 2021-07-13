"""Tools for vizualisation of convolutional neural network filter in keras models."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools


def to_rec(box, image_size):
    """Finds minimum rectangle around some points and scales it to desired 
    image size.
    
    # Arguments
        box: Box or points [x1, y1, x2, y2, ...] with values between 0 and 1.
        image_size: Size of output image.
    # Return
        xy_rec: Corner coordinates of rectangle, array of shape (4, 2).
    """
    image_h, image_w = image_size
    xmin = np.min(box[0::2]) * image_w
    xmax = np.max(box[0::2]) * image_w
    ymin = np.min(box[1::2]) * image_h
    ymax = np.max(box[1::2]) * image_h
    xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return xy_rec


def plot_box(box, box_format='xywh', color='r', linewidth=1, normalized=False, vertices=False):
    if box_format == 'xywh': # opencv
        xmin, ymin, w, h = box
        xmax, ymax = xmin + w, ymin + h
    elif box_format == 'xyxy':
        xmin, ymin, xmax, ymax = box
    if box_format == 'polygon':
        xy_rec = np.reshape(box, (-1, 2))
    else:
        xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    if normalized:
        im = plt.gci()
        xy_rec = xy_rec * np.tile(im.get_size(), (4,1))
    ax = plt.gca()
    ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=linewidth))
    if vertices:
        c = 'rgby'
        for i in range(4):
            plt.plot(xy_rec[i,0],xy_rec[i,1], c[i], marker='o', markersize=4)

