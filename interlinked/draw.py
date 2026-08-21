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
    lbls = np.unique(rois)
    lbls = lbls[lbls != bkgd]
    Ln = len(lbls)

    h = np.random.rand(Ln)
    hsv = np.stack([h, np.ones(Ln), np.ones(Ln)], axis=1)
    rgb = hsv_to_rgb(hsv)
    rgb = np.rint(rgb * 255).astype(np.uint8)

    background = np.zeros((1, 3), np.uint8)
    lut = np.vstack([rgb, background])

    new = np.searchsorted(lbls, rois)
    new[rois != bkgd] = Ln
    rois = lut[new]
    return rois


