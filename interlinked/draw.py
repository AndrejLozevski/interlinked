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
BLUE_OVERLAY = LinearSegmentedColormap("green_overlay", {
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


#--| Functions |------------------------------------------------------------------------#

# Draws colored ROIs in a volume
def rois(rois, shape):
    Lz, Ly, Lx = shape
    hsv = np.zeros((rois.max() + 1, 3), np.float32)
    hsv[1:, 0] = np.random.rand(rois.max() + 1)
    hsv[1:, 1] = 1.0
    hsv[1:, 2] = 1.0

    rgb = hsv_to_rgb(hsv)
    rgb = np.rint(rgb * 255).astype(np.uint8)
    color = rgb[rois + 1]
    rois = color.transpose(0,3,1,2)
    return rois


