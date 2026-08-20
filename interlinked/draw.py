import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap


#--| Constants |------------------------------------------------------------------------#

RED_OVERLAY = LinearSegmentedColormap("red_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})
GREEN_OVERLAY = LinearSegmentedColormap("green_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})
BLUE_OVERLAY = LinearSegmentedColormap("blue_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})

MAGENTA_OVERLAY = LinearSegmentedColormap("magenta_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})
CYAN_OVERLAY = LinearSegmentedColormap("cyan_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})
YELLOW_OVERLAY = LinearSegmentedColormap("yellow_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})

WHITE_OVERLAY = LinearSegmentedColormap("white_overlay", {
    "red":   ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "blue":  ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    "alpha": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0))  
})


#--| Functions |------------------------------------------------------------------------#

# Draws colored ROIs in a 2D array
def color_rois(rois, bkgd=-1):
    Ly, Lx = rois.shape
    img = np.zeros((3, Ly, Lx), np.uint8)
    for i in np.unique(rois):
        if i == bkgd:
            continue
        h = np.random.rand(1)[0]
        hsv = [h, 1.0, 1.0]
        rgb = hsv_to_rgb(hsv)
        rgb = np.rint(rgb * 255).astype(np.uint8)
        img[:, rois == i] = rgb[:, None]
    img = img.transpose(1,2,0)
    return img


