import os
import sys
import logging
import numpy as np
import scipy as sp
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)


# Formats the tqdm progress bar to match logging format
def format_log(desc, tag='INFO'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
    return f'{timestamp} [{tag}] - {desc}'

# Digitizes data into n bins
def digitize(data, n, dtype=np.int32):
    bins = np.linspace(data.min(), data.max(), n+1)
    data = np.digitize(data, bins)
    return data.astype(dtype)

# Calculates dF/F from raw data
def dff(raw, downsample=1, percentile=20, window=300):
    if downsample == 1:
        baseline = sp.ndimage.filters.percentile_filter(raw, percentile=percentile, size=window)
    else:
        decimated = sp.signal.decimate(raw, downsample, ftype='iir', zero_phase=True)
        decimated += raw.min() - decimated.min()
        baseline = sp.ndimage.filters.percentile_filter(decimated, percentile=percentile, size=window//downsample)

    baseline = np.interp(range(0, len(raw)), range(0, len(raw), downsample), baseline)
    assert len(baseline) == len(raw)
    return (raw - baseline) / (baseline - baseline.min() + 1)

# Safely converts a numpy array into a safe divisor
def divisor(arr, minimum=1, default_positive=True):
    default_sign = 1 if default_positive else -1
    signs = np.sign(arr)
    signs[signs == 0] = default_sign
    return signs * np.maximum(np.abs(arr), minimum)

# Convolves a time series with an exponential decay function
def decay(data, tau=2.00, width=16, inv=False):
    krnl = tau ** np.arange(width)
    krnl = np.exp(-np.arange(width) / tau)
    if inv:
        return np.convolve(data[::-1], krnl, mode='full')[:len(data)][::-1]
    return np.convolve(data, krnl, mode='full')[:len(data)]

def exponential_decay(Lt, A, tau, C):
    return A * np.exp(-Lt / tau) + C    




