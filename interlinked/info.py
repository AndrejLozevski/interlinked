import logging
import numpy as np
import interlinked as lnk

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

NUM_BINS = lnk.config.NUM_BINS


#--| Utilities |------------------------------------------------------------------------#

# Creates usable bins tuple
def _bins(path, new=False):
    if isinstance(bins, int):
        bins = (bins, bins) if num == 2 else (bins, bins, bins)
    elif isinstance(bins, (tuple, list)):
        if len(bins) != num:
            log.error('Must provide 1 bin count or a list of %s bin counts', num)
            sys.exit(1)
    if num == 2:
        bins = np.minimum(bins, (len(np.unique(x)), len(np.unique(y))))
    elif num == 3:
        assert z is not None
        bins = np.minimum(bins, (len(np.unique(x)), len(np.unique(y)), len(np.unique(z))))
    else:
        log.error('Invalid variable count: %s', num)
        sys.exit(1)
    return bins


#--| Entropy |--------------------------------------------------------------------------#

# Calculates entropy of a histogram
def hist_H(counts):
    p = counts / np.sum(counts)
    p = np.clip(p, 1e-10, None)
    return -np.sum(p * np.log2(p))


#--| Discrete MI |----------------------------------------------------------------------#

# Calculates Mutual Information between two discrete variables
def disc_MI(x, y, normalized=False, bins=NUM_BINS):
    bins = _bins(bins, 2, x, y)

    hist_xy = np.histogram2d(x, y, bins=(bins[0], bins[1]))[0]
    hist_x  = np.histogram(x, bins=bins[0])[0]
    hist_y  = np.histogram(y, bins=bins[1])[0]

    Hxy = hist_H(hist_xy)
    Hx  = hist_H(hist_x)
    Hy  = hist_H(hist_y)

    MI = Hx + Hy - Hxy
    MI = np.clip(MI, 0.0, None)
    if normalized:
        den = np.sqrt(Hx * Hy)
        if den == 0:
            return 0.0
        return MI / den
    return MI

# Calculates Conditional Mutual Information between two discrete variables
def discrete_cMI(x, y, z, normalized=False, bins=NUM_BINS):
    bins = _bins(bins, 3, x, y, z)

    data = np.column_stack((x, y, z))
    hist_xyz = np.histogramdd(data, bins=(bins[0], bins[1], bins[2]))[0]
    hist_xz  = np.histogram2d(x, z, bins=(bins[0], bins[2]))[0]
    hist_yz  = np.histogram2d(y, z, bins=(bins[1], bins[2]))[0]
    hist_z   = np.histogram(z, bins=bins[2])[0]

    Hxyz = hist_H(hist_xyz)
    Hxz  = hist_H(hist_xz)
    Hyz  = hist_H(hist_yz)
    Hz   = hist_H(hist_z)

    cMI = Hxz + Hyz - Hxyz - Hz
    cMI = np.clip(cMI, 0.0, None)
    if normalized:
        Hx_z = Hxz - Hz
        Hy_y = Hyz - Hz
        den = np.sqrt(Hx_z * Hy_z)
        if den == 0:
            return 0.0
        return cMI / den
    return cMI

# Calculates Interaction Information between three discrete variables
def discrete_iMI(x, y, z, normalized=False, bins=NUM_BINS):
    bins = _bins(bins, 3, x, y, z)

    data = np.column_stack((x, y, z))
    hist_xyz = np.histogramdd(data, bins=(bins[0], bins[1], bins[2]))[0]
    hist_xy  = np.histogram2d(x, y, bins=(bins[0], bins[1]))[0]
    hist_xz  = np.histogram2d(x, z, bins=(bins[0], bins[2]))[0]
    hist_yz  = np.histogram2d(y, z, bins=(bins[1], bins[2]))[0]
    hist_x   = np.histogram(x, bins=bins[0])[0]
    hist_y   = np.histogram(y, bins=bins[1])[0]
    hist_z   = np.histogram(z, bins=bins[2])[0]

    Hxyz = hist_H(hist_xyz)
    Hxy  = hist_H(hist_xy)
    Hxz  = hist_H(hist_xz)
    Hyz  = hist_H(hist_yz)
    Hx   = hist_H(hist_x)
    Hy   = hist_H(hist_y)
    Hz   = hist_H(hist_z)

    iMI = (Hx + Hy + Hz) - (Hxy + Hxz + Hyz) + Hxyz
    if normalized:
        den = min(Hx, Hy, Hz)
        if den == 0:
            return 0.0
        return iMI / den
    return iMI


#--| Continuous MI |--------------------------------------------------------------------#

# TODO: Convert old Numba code into Rust code and reference here



