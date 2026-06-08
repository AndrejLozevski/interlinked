import os
import sys
import logging
import numpy as np
import scipy as sp
from pathlib import Path
from datetime import datetime

import interlinked as lnk

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

LOG_LEVEL    = "INFO"
LOG_DATETIME = "%Y-%m-%d %H:%M:%S,%f"

#--| Functions |------------------------------------------------------------------------#

# Formats the tqdm progress bar to match logging format
def format_log(desc, tag=LOG_LEVEL):
    timestamp = datetime.now().strftime(LOG_DATETIME)[:-3]
    return f"{timestamp} [{tag}] - {desc}"

# Digitizes data into n bins
def digitize(data, n, dtype=np.int32):
    bins = np.linspace(data.min(), data.max(), n+1)
    data = np.digitize(data, bins)
    return data.astype(dtype)

# Interpolates with vectorization
def interpolate(x, xp, fp):
    x  = np.asarray(x,  dtype=np.float32)
    xp = np.asarray(xp, dtype=np.float32)
    fp = np.asarray(xp, dtype=np.float32)

    idx = np.searchsorted(xp, x, side='right') - 1
    idx = np.clip(idx, 0, len(xp) - 2)

    x_lo = xp[idx]
    x_hi = xp[idx+1]

    f_lo = fp[:,idx]
    f_hi = fp[:,idx+1]

    t = (x - x_lo) / (x_hi - x_lo)
    t = np.clip(t, 0, 1)
    return f_lo + t * (f_hi - f_lo)

# Safely converts a numpy array into a safe divisor
def divisor(arr, minimum=1, default_positive=True):
    default_sign = 1 if default_positive else -1
    signs = np.sign(arr)
    signs[signs == 0] = default_sign
    return signs * np.maximum(np.abs(arr), minimum)

# Calculates dF/F from raw data
def dff(raw, downsample=1, percentile=20, window=300):
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)

    if downsample == 1:
        baseline = sp.ndimage.percentile_filter(raw, percentile=percentile, size=(1, window))
    else:
        decimated = sp.signal.decimate(raw, downsample, axis=-1, ftype="iir", zero_phase=True)
        decimated += raw.min() - decimated.min()
        baseline = sp.ndimage.percentile_filter(decimated, percentile=percentile, size=(1, window//downsample))

    Lc, Lt = raw.shape
    baseline = interpolate(range(0, Lt), range(0, Lt, downsample), baseline)
    assert len(baseline) == len(raw)
    return (raw - baseline) / np.abs(divisor(baseline))

# Convolves a time series with an exponential decay function
def decay(data, tau=2.00, width=16, inv=False):
    krnl = tau ** np.arange(width)
    krnl = np.exp(-np.arange(width) / tau)
    if inv:
        return np.convolve(data[::-1], krnl, mode="full")[:len(data)][::-1]
    return np.convolve(data, krnl, mode="full")[:len(data)]

def exponential_decay(Lt, A, tau, C):
    return A * np.exp(-Lt / tau) + C    




