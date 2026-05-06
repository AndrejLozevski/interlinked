import os
import sys
import uuid
import h5py
import logging
import numpy as np
import tifffile as tiff
import skimage.morphology as morph
import xml.etree.ElementTree as ET

from pathlib import Path
from dataclasses import dataclass, field

import interlinked as lnk

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

TIF2D = {"imagej":True, "metadata":{"axes": "YX"}}
TIF3D = {"imagej":True, "metadata":{"axes":"ZYX"}}


#--| Utilities |------------------------------------------------------------------------#

# Ensures a specified path is a Path instance
def _path(path, new=False):
    if type(path) == str:
        path = Path(path)
    if (not new) and (not path.exists()):
        lnk.meta.Error("Path '%s' does not exist", path, error=FileNotFoundError)
        sys.exit(1)
    return path

# finds a file
def find_file(path, pattern):
    path = _path(path)

    files = list(path.glob(pattern))
    if len(files) == 0:
        lnk.meta.Error("No '%s' file found in '%s'", pattern, path, error=FileNotFoundError)
        sys.exit(1)
    if len(files) > 1:
        lnk.meta.Error("Found %s file(s) with the pattern '%s' in '%s'", len(files), pattern, path)
        sys.exit(1)

    file = files[0]
    return file

# Loads a file
def load_file(path, pattern, pickled=False):
    path = _path(path)
    file = find_file(path, pattern)

    if file.suffix == ".npy":
        data = np.load(file, allow_pickle=pickled)
    elif file.suffix == ".tif":
        data = tiff.imread(file)
    elif file.suffix in [".h5", ".hdf5"]:
        data = h5py.File(file, "r")
    else:
        lnk.meta.Error("Unrecognized file extension: '%s'", file.suffix)
        sys.exit(1)
    return data

# Creates temporary filename (for memmap files)
def temp_filepath():
    temp_id = uuid.uuid4()
    filepath = lnk.config.TEMP_DIR / f"{lnk.config.TEMP_PREFIX}{temp_id}{lnk.config.TEMP_SUFFIX}"
    return filepath

# Safely deletes temporary memmap files from the temp directory (VERY STRICT)
def safely_delete(path):
    path = _path(path)
    temp_dir = lnk.config.TEMP_DIR
    prefix = lnk.config.TEMP_PREFIX
    suffix = lnk.config.TEMP_SUFFIX

    if not path.parent == temp_dir:
        lnk.meta.Error("File directory '%s' doesn\'t match the /temp directory '%s'", path.parent, temp_dir)
        sys.exit(1)
    elif not str(path.name).startswith(prefix):
        lnk.meta.Error("File name '%s' doesn\'t start with the expected prefix '%s'", path.parent, temp_dir)
        sys.exit(1)
    elif not path.suffix == suffix:
        lnk.meta.Error("File extension '%s' doesn\'t match the expected suffix '%s'", path.suffix, suffix)
        sys.exit(1)
    else:
        os.remove(path)
    return

# Checks for leftover temporary files
def check_temp(clear=False):
    files = list(lnk.config.TEMP_DIR.iterdir())
    if len(files) > 0:
        if clear:
            clear_temp(False)
            log.warning("Found and deleted %s temporary file(s) in temp directory.", len(files))
        else:
            log.warning("Found %s temporary file(s) in temp directory. Delete them if not in use.", len(files))
    return

# Clears leftover temporary files
def clear_temp(notify=True):
    files = list(lnk.config.TEMP_DIR.iterdir())
    if len(files) > 0:
        for file in files:
            safely_delete(file)
        if notify:
            log.info("Temp directory cleared of %s temporary file(s).", len(files))
    return

        
#--| Memmaps |--------------------------------------------------------------------------#

# Stores memmap metadata
@dataclass
class Memmap:
    shape: tuple
    dtype: str
    path:  Path = field(default_factory=temp_filepath)

    def save(self, data):
        mem = np.memmap(self.path, shape=self.shape, dtype=self.dtype, mode="w+")
        mem[:] = data
        return

    def load(self, read_only=True):
        mode = "r" if read_only else "r+"
        mem = np.memmap(self.path, shape=self.shape, dtype=self.dtype, mode=mode)
        return mem

    def delete(self):
        safely_delete(self.path, True)
        return


#--| Metadata |-------------------------------------------------------------------------#

# Loads fps from parent directory
def load_fps(path):
    path = _path(path)
    fps_file = find_file(path, "*frequency.txt")
    with open(fps_file, "r") as file:
        fps = float(file.readline().strip())
    return fps

# Loads resolution from parent directory
def load_resolution(path):
    path = _path(path)
    file = find_file(path, "*.xml")

    Ry, Rx = (0.406, 0.406)
    data = ET.parse(file)
    root = data.getroot()
    for info in root.findall("info"):
        if (Rz := info.get("z_step")) is not None:
            return float(Rz), Ry, Rx
    else:
        lnk.meta.Error("No 'z_step' tag found in '%s'", file.name, error=KeyError)
        sys.exit(1)
    return

# Loads metadata form parent directory
def load_metadata(path):
    path = _path(path)
    Rz, Ry, Rx = load_resolution(path)
    Rt = 1 / load_fps(path)
    return Rz, Ry, Rx, Rt


#--| Suite2p Data |---------------------------------------------------------------------#

# Ensures that the provided path is the Suite2p combined directory
def ensure_suite2p_path(path):
    path = _path(path)
    if (path / "suite2p").exists():
        path = path / "suite2p"
    if (path / "combined").exists():
        path = path / "combined"
    if path.name != "combined":
        lnk.meta.Error("No suite2p combined directory found at %s", path, error=FileNotFoundError)
        sys.exit(1)

    needed_files = ["stat.npy", "F.npy", "ops.npy"]
    for file in needed_files:
        if path / file not in list(path.iterdir()):
            lnk.meta.Error("file '%s' not found in %s", file, path, error=FileNotFoundError)
            sys.exit(1)
    return path

# Aligns ROIs into a volume from a Suite2p combined stat.npy file
def align_rois(cell_locations, shape):
    Lc, Lt, Lz, Ly, Lx = shape
    cell_id = 0
    rois = -1 * np.ones((Lz, Ly, Lx), dtype=np.int64)
    for c in range(Lc):
        ypix   = cell_locations[c]["ypix"]
        xpix   = cell_locations[c]["xpix"]
        zplane = cell_locations[c]["iplane"]
        assert len(ypix) == len(xpix)

        for i in range(len(ypix)):
            rois[zplane, divmod(ypix[i], Ly)[1], divmod(xpix[i], Lx)[1]] = cell_id
        cell_id += 1
    return rois

# Loads brainmap as a volume from ops.npy file
def load_brainmap(ops, shape):
    Lc, Lt, Lz, Ly, Lx = shape
    mean = ops["meanImg"].astype(np.float32)
    bmap = lnk.form_volume(mean, (Lz, Ly, Lx))

    Oy, Ox = mean.shape
    assert (Oy//Ly) * (Ox//Lx) >= Lz

    z = 0
    bmap = np.empty((Lz, Ly, Lx), np.float32)
    for i in range(Oy//Ly):
        for j in range(Ox//Lx):
            if z >= Lz:
                break
            bmap[z,:,:] = mean[i*Ly:(i+1)*Ly, j*Lx:(j+1)*Lx]
            z += 1
    assert bmap.dtype == np.float32 and bmap.shape == (Lz, Ly, Lx)
    return bmap

# Loads Suite2p data from the given path
def load_suite2p_data(path):
    path = ensure_suite2p_path(path)
    cell_locations = np.load(path / "stat.npy", allow_pickle=True)
    cell_traces    = np.load(path / "F.npy",    allow_pickle=True)
    ops            = np.load(path / "ops.npy",  allow_pickle=True).item()

    baseline = np.percentile(cell_traces, 20, axis=1, keepdims=True)
    cell_traces = (cell_traces - baseline) / np.abs(lnk.utils.divisor(baseline))
    cell_traces = (cell_traces - cell_traces.mean()) / cell_traces.std()

    shape = (
        cell_traces.shape[0],    # Lc, cell count
        cell_traces.shape[1],    # Lt, timepoints
        ops["nplanes"],          # Lz, length Z
        ops["lenY"],             # Ly, length Y
        ops["lenX"]              # Lx, length X
    )

    bmap = load_brainmap(ops, shape)
    rois = align_rois(cell_locations, shape)
    return roise, cell_traces, bmap, shape, ops


#--| VoluSeg Data |---------------------------------------------------------------------#

# Ensures that the provided path is the VoluSeg directory
def ensure_voluseg_path(path):
    path = _path(path)
    if (path / "voluseg").exists():
        path = path / "voluseg"
    if path.name != "voluseg":
        lnk.meta.Error("No voluseg directory found at %s", path, error=FileNotFoundError)
        sys.exit(1)

    needed_files = ["cells0_clean.hdf5", "volume0.hdf5"]
    for file in needed_files:
        if path / file not in list(path.iterdir()):
            lnk.meta.Error("File '%s' not found in %s", file, path, error=FileNotFoundError)
            sys.exit(1)
    return path

# Loads VoluSeg data from the given path
def load_voluseg_data(path):
    path = ensure_voluseg_path(path)

    volume_data = h5py.File(path / "volume0.hdf5",      "r")
    cell_data   = h5py.File(path / "cells0_clean.hdf5", "r")

    raw_traces = cell_data["cell_timeseries"][:].astype(np.float32)
    baseline   = cell_data["cell_baseline"][:].astype(np.float32)
    cell_traces = (raw_traces - baseline) / np.abs(lnk.utils.divisor(baseline))
    cell_traces = (cell_traces - cell_traces.mean()) / cell_traces.std()
    assert raw_traces.shape == cell_traces.shape == baseline.shape

    bmap = volume_data["volume_mean"][:].astype(np.float32)
    rois = cell_data["volume_id"][:].T.astype(np.float32)

    shape = (
        cell_traces.shape[0],    # Lc, cell count
        cell_traces.shape[1],    # Lt, timepoints
        bmap.shape[0],           # Lz, length Z
        bmap.shape[1],           # Ly, length Y
        bmap.shape[2]            # Lx, length X
    )
    return rois, cell_traces, bmap, shape


#--| Combined Data |---------------------------------------------------------------------#

# Ensures that the provided path is the Suite2p combined directory
def ensure_combined_path(path):
    path = _path(path)
    needed_files = ["combined_segdata.h5"]
    for file in needed_files:
        if path / file not in list(path.iterdir()):
            lnk.meta.Error("File '%s' not found in %s", file, path, error=FileNotFoundError)
            sys.exit(1)
    return path

# Loads VoluSeg data from the given path
def load_combined_data(path, metadata=False):
    path = ensure_combined_path(path)

    with h5py.File(path / "combined_segdata.h5", "r") as file:
        rois = file["rois"][:]
        cell_traces = file["traces"][:]
        bmap = file["bmap"][:]
        shape = file["shape"][:]
        if metadata:
            metadata = file["metadata"][:]
            return rois, cell_traces, bmap, shape, metadata
        return rois, cell_traces, bmap, shape


#--| Trials |---------------------------------------------------------------------------#

# Creates a (trial, time) array of time indices for masking
def build_trials(drift):
    mask_drift = (drift != 0)
    mask_move  = morph.remove_small_objects(mask_drift, min_size=10)
    mask_wait  = ~mask_move
    mask_pulse = mask_wait & mask_drift

    def measure_periods(mask):
        labels = morph.label(mask)
        sums = [
            np.sum(labels == i) for i in range(1, labels.max() + 1)
        ]
        counts = [
            (i, sum([sums[j] == i for j in range(len(sums))]))
            for
            i in np.unique(sums)
        ]
        mask_sort = sorted(counts, key=lambda count: count[1])
        mask_len  = mask_sort[-1][0]
        mask_num  = mask_sort[-1][1]
        return mask_len, mask_num

    move_len, move_num = measure_periods(mask_move)
    wait_len, wait_num = measure_periods(mask_wait)
    
    Ltt = wait_len + move_len
    Lt = len(drift)
    Ln = min(move_num, wait_num)

    starts = np.nonzero(np.diff(mask_move.astype(np.uint8)) == 1)[0] + 1
    periods = []
    for i in range(len(starts)):
        if i == len(starts) - 1:
            if Lt - starts[i] >= Ltt:
                periods.append(starts[i] + np.arange(Ltt))
            continue
        
        length = starts[i+1] - starts[i]
        if length != Ltt:
            continue
        periods.append(starts[i] + np.arange(Ltt))
    
    Ln = len(periods)
    trials = np.zeros((Ln, Ltt), np.uint16)
    for n in range(Ln):
        trials[n,:] = periods[n]

    pulses = mask_pulse[trials]
    pulse  = np.nonzero(pulses.sum(axis=0))[0]
    pulses[:, pulse] = True
    return trials, mask_move[trials], mask_wait[trials], pulses 


