import logging
import numpy as np
import interlinked as lnk

from .interlinked import info as rs_info

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

BIN_TYPE = 'fixed'
NUM_BINS = lnk.config.NUM_BINS
NUM_KNNS = lnk.config.NUM_KNNS

EPSILON = 1e-10

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
    p = np.clip(p, EPSILON, None)
    return -np.sum(p * np.log2(p))

# Calculates entropy of a continous variable using Kozachenko-Leonenko estimator
def KL_H(x, k=NUM_KNNS):
    x = np.asarray(x, dtype=np.float64)
    return rs_info.kl_h(x, k)


#--| Discrete MI |----------------------------------------------------------------------#

# Calculates Mutual Information between two discrete variables
def disc_MI(x, y, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE):
    n_bins = _bins(n_bins, 2, x, y)

    if bin_type == 'fixed':
        bins_x, bins_y = n_bins
    elif bin_type == 'quantile':
        bins_x = lnk.stats.quantile_bins(x, n_bins[0])        
        bins_y = lnk.stats.quantile_bins(y, n_bins[1])        
    else:
        log.error("Invalid bin type: '%s'", bin_type)
        sys.exit(1)

    hist_xy = np.histogram2d(x, y, bins=(bins_x, bins_y))[0]
    hist_x  = np.sum(hist_xy, axis=0)
    hist_y  = np.sum(hist_xy, axis=1)

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
def discrete_cMI(x, y, z, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE):
    n_bins = _bins(n_bins, 3, x, y, z)
    data = np.column_stack((x, y, z))

    if bin_type == 'fixed':
        bins_x, bins_y, bins_z = n_bins
    elif bin_type == 'quantile':
        bins_x = lnk.stats.quantile_bins(x, n_bins[0])        
        bins_y = lnk.stats.quantile_bins(y, n_bins[1])        
        bins_z = lnk.stats.quantile_bins(z, n_bins[2])        
    else:
        log.error("Invalid bin type: '%s'", bin_type)
        sys.exit(1)

    hist_xyz = np.histogramdd(data, bins=(bins_x, bins_y, bins_z))[0]
    hist_xz  = np.sum(hist_xyz, axis=1)
    hist_yz  = np.sum(hist_xyz, axis=0)
    hist_z   = np.sum(hist_xz,  axis=0)

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
def discrete_iMI(x, y, z, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE):
    n_bins = _bins(n_bins, 3, x, y, z)
    data = np.column_stack((x, y, z))

    if bin_type == 'fixed':
        bins_x, bins_y, bins_z = n_bins
    elif bin_type == 'quantile':
        bins_x = lnk.stats.quantile_bins(x, n_bins[0])        
        bins_y = lnk.stats.quantile_bins(y, n_bins[1])        
        bins_z = lnk.stats.quantile_bins(z, n_bins[2])        
    else:
        log.error("Invalid bin type: '%s'", bin_type)
        sys.exit(1)

    hist_xyz = np.histogramdd(data, bins=(bins_x, bins_y, bins_z))[0]
    hist_xy  = np.sum(hist_xyz, axis=2)
    hist_xz  = np.sum(hist_xyz, axis=1)
    hist_yz  = np.sum(hist_xyz, axis=0)
    hist_x   = np.sum(hist_xy,  axis=1)
    hist_y   = np.sum(hist_xy,  axis=0)
    hist_z   = np.sum(hist_xz,  axis=0)

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

# Calculates MI of two continous variables using Kraksov-Stogbauer-Grassberger estimator
def KSG_MI(x, y, k=NUM_KNNS): 
    x = x.asarray(x, dtype=np.float64)
    y = y.asarray(y, dtype=np.float64)
    return rs_info.ksg_mi(x, y, k)

# Calculates conditional MI of two continous variables given a third variable using Kraksov-Stogbauer-Grassberger estimator
def KSG_CMI(x, y, z, k=NUM_KNNS): 
    x = x.asarray(x, dtype=np.float64)
    y = y.asarray(y, dtype=np.float64)
    z = z.asarray(z, dtype=np.float64)
    return rs_info.ksg_cmi(x, y, z, k)

# Calculates interaction information of three continous variables using Kraksov-Stogbauer-Grassberger estimator
def KSG_CMI(x, y, z, k=NUM_KNNS): 
    x = x.asarray(x, dtype=np.float64)
    y = y.asarray(y, dtype=np.float64)
    z = z.asarray(z, dtype=np.float64)
    return rs_info.ksg_ii(x, y, z, k)


