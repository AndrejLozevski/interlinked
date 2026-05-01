import sys
import cv2
import logging
import numpy as np
import scipy as sp
import skimage.morphology as morph

import interlinked as lnk

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

ROW_TILES = 5

#--| Reshaping |------------------------------------------------------------------------#

# Forms a volume from a tiled image
def form_volume(img, shape):
    Lz, Ly, Lx = shape
    img = img.astype(np.float32)
    Oy, Ox = img.shape
    assert (Oy//Ly) * (Ox//Lx) >= Lz

    z = 0
    volume = np.empty(shape, np.float32)
    for iy in range(Oy//Ly):
        for ix in range(Ox//Lx):
            if z >= Lz:
                break
            volume[z,:,:] = img[iy*Ly:(iy+1)*Ly, ix*Lx:(ix+1)*Lx]
            z += 1
    assert volume.shape == (Lz, Ly, Lx)
    return volume

# Forms a tiled image from a volume
def form_tiles(vol, shape):
    Lz, Ly, Lx = shape
    Tx = ROW_TILES
    q, r = divmod(Lz, Tx)
    Ty = q if r == 0 else q + 1

    z = 0
    img = np.ones((Ty*Ly, Tx*Lx), vol.dtype) * vol.min()
    for iy in range(Ty):
        for ix in range(Tx):
            if z >= Lz:
                break
            img[iy*Ly:(iy+1)*Ly, ix*Lx:(ix+1)*Lx] = vol[z,:,:]
            z += 1
    assert img.shape == (Ty*Ly, Tx*Lx)
    return img


#--| ROIs |-----------------------------------------------------------------------------#

# Labels ROIs in a volume from a Suite2p combined stat.npy file
def label_rois(stat, shape):
    Lz, Ly, Lx = shape
    Ln = len(stat)

    count = 0
    rois = np.zeros((Lz,Ly,Lx), np.int32)
    for n in range(Ln):
        ypix = stat[n]['ypix']    
        xpix = stat[n]['xpix']    
        zpln = stat[n]['iplane']    
        assert len(ypix) == len(xpix)

        for py, px in zip(ypix, xpix):
            rois[zpln, py%Ly, px%Lx] = count + 1
        count += 1
    return rois

# Shifts ROIs in a labeled array to fill any missing label IDs
def adjust_rois(arr):
    uniq = np.unique(arr)
    uniq = uniq[uniq != uniq.min()]
    if uniq.size == 0:
        log.error('No labels found in array')
        sys.exit(1)

    lmin = uniq.min()
    lmax = uniq.max()

    lut = np.arange(lmax - lmin + 1, dtype=arr.dtype)
    lut[:] = -999
    lut[uniq - lmin] = np.arange(len(uniq)) + lmin
    
    out = arr.copy()
    mask = (arr >= lmin) & (arr <= lmax)
    out[mask] = lut[arr[mask] - lmin]

    full = np.arange(lmin, lmax+1)
    mask = np.in1d(full, uniq, assume_unique=True)
    missing = full[~mask].tolist()
    return out, uniq, missing

# Removes the specified ROIs from a labeled array
def remove_rois(arr, rois, keep=False):
    if keep:
        keep = np.array(rois)
        mask = np.isin(arr, rois)
        arr[~mask] = arr.min()
    else:
        lookup = np.zeros(arr.max()+1, bool)
        lookup[rois] = True
        arr[lookup[arr]] = arr.min()
    return arr

# Ensures ROIs in a labeled array are valid and without gaps
def validate_rois(rois, Lc=0):
    if not rois.min() == -1:
        log.error("Background label should be -1, but found: (Min: %s)", rois.min())
        sys.exit(1)
    if not len(np.unique(rois)) == rois.max()+2:
        log.error("Unique label count should be two more than max label, but found: (Unique: %s, Max: %s)", len(np.unique(rois)), rois.max())
        sys.exit(1)
    if Lc != 0:
        if not rois.max() == Lc-1:
            log.error("Max label should be one more than roi count, but found: (Max: %s, Lc: %s)", rois.max(), Lc)
            sys.exit(1)
        if not rois.max() - rois.min() == Lc:
            log.error("Max label minus min label should equal roi count, but found: (Max: %s, Min: %s, Lc: %s)", rois.max(), rois.min(), Lc)
            sys.exit(1)
        if not len(np.unique(rois)) == Lc+1:
            log.error("Unique label count should one more than roi count, but found: (Max: %s, Min: %s, Lc: %s)", rois.max(), rois.min(), Lc)
            sys.exit(1)
    return







