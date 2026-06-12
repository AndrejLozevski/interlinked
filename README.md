# The Interlinked Module
This module is for neuroscientific computation and analysis with Python submodules and Rust-optimized kernels.
This code is developed by Andrej Lozevski for the En Yang Lab at UNC Chapel Hill (2026).


## Installation
This module can be installed by running:
```bash
pip install interlinked-lab
```

## Submodules
All code is organized into the following submodules:
- **utils** --- Contains functions for a wide range of uses
- **io** --- Contains functions for I/O operations and streamlined file use
- **form** --- Contains functions for manipulating labeled and non-labeled arrays
- **stats** --- Contains functions for statistical tests and measures
- **info** --- Contains functions for information theoretic tests and measures
- **draw** --- Contains functions for data visualization
- **config** --- Contains methods for updating, resetting, loading, and saving parameter settings for the module's default behaviors and arguments

Each submodule can be used in a program directly or through imports:
```python
# Method 1
import interlinked
print(interlinked.config.defaults())

# Method 2
from interlinked import config
print(config.defaults())
```

All submodules and their functions/methods are listed below[^1]:
[^1]: Only functions built for typical users are shown below. Additional functions not shown here are considered peripheral and should only be used after reading the source code.
---

### Config<br>
This submodule is a class instance, possessing the following methods and attributes:

#### Methods<br>
- `def defaults(self)`<br>
  **Lists all config default parameters**<br>
  returns: *None*<br>

- `def list(self)`<br>
  **Lists all current config parameters**<br>
  returns: *None*<br>

- `def configure(self, **kwargs)`<br>
  **Updates any specified config parameters**<br>
  *Default parameters cannot be changed to a different type, but custom parameters have no restrictions*<br>
  returns: *None*<br>

- `def reset(self)`<br>
  **Resets config parameters to defaults**<br>
  returns: *None*<br>

- `def save(self)`<br>
  **Saves current config parameters in the temp directory**<br>
  *Overwrites any pre-existing saved config parameters*<br>
  returns: *None*<br>

- `def load(self)`<br>
  **Loads config parameters saved in the temp directory**<br>
  *If no saved parameters are found, throws an Exception and loads defaults*<br>
  returns: *None*<br>

#### Attributes<br>
- `TEMP_DIR:` *Path = /tmp/interlinked*<br>
  where the temp directory is located **(DO NOT CHANGE)**<br>
- `TEMP_PREFIX:` *str = "\_\_temp\_\_"*<br>
  file prefix given to temporary files **(DO NOT CHANGE)**<br>
- `TEMP_SUFFIX:` *str = ".dat"*<br>
  file suffix given to temporary files **(DO NOT CHANGE)**<br>
- `CLEAR_TEMP:` *bool = False*<br>
  *see io.check_temp() below*

- `BATCH_SIZE:` *int = 1000*<br>
  size of batches for parallelism<br>
- `NUM_WORKERS:` *int = 8*<br>
  number of workers for parallelism<br>

- `NUM_BINS:` *int = 5*<br>
  number of bins for binning and digitization<br>
- `NUM_KNNS:` *int = 8*<br>
  number of nearest neighbors for KNN operations<br>
- `NUM_ITER:` *int = 1_000_000*<br>
  number of iterations for monte carlo simulation and bootstrap resampling operations<br>

- `ALPHA1:` *float = 0.05*<br>
  α₁ for statistical tests<br>
- `ALPHA2:` *float = 0.01*<br>
  α₂ for statistical tests<br>
- `ALPHA3:` *float = 0.001*<br>
  α₃ for statistical tests<br>
- `ALPHA4:` *float = 0.0001*<br>
  α₄ for statistical tests<br>

- `RADIUS:` *int*<br>
  radius used for graph theory and nearest neighbors operations<br>
- `MIN_SIZE:` *int*<br>
  minimum node count for graph theory operations<br>

Any ALL-CAPS variable listed below can be assumed as a parameter found in the default config instance.
---

### Utils 
- `def digitize(x, n, dtype=np.int32)`<br>
  **Digitizes an input array into discretized values from a given number of fixed-width bins**<br>
  x: *ndarray (ndim of 1)* --- input array to digitize<br>
  n: *int* --- desired range of the output array<br>
  dtype: *dtype* --- target dtype of output array<br>
  returns: *ndarray (ndim of 1)*<br>

- `def interpolate(x, xp, fp)`<br>
  **Interpolates an input array with vectorized calculation**<br>
  x: *ndarray (ndim of 1)* --- x-values of target data<br>
  xp: *ndarray (ndim of 1)* --- x-values of input data<br>
  fp: *ndarray (ndim of 2)* --- y-values of input data<br>
  returns: *ndarray (ndim of 2)*<br>
    
- `def dff(raw, downsample=1, percentile=20, window=300)`<br>
  **Calculates the ΔF/F of a calcium trace using a percentile filter and a sliding window**<br>
  raw: *ndarray (ndim of 1 or 2)* --- input array for which to calculate ΔF/F, assuming shape (cells, timepoints)<br>
  downsample: *int* --- downsampling factor (setting to 1 prevents downsampling)<br>
  percentile: *float* --- percentile with which to calculate the baseline of the time series<br>
  window: *int* --- sliding window size with which to calculate the baseline of the time series<br>
  returns: *ndarray (ndim of raw)*<br>

- `def divisor(arr, minimum=1, default_positive=True)`<br>
  **Converts an input array into a safe divisor for array division, keeping sign and preventing unintentional mulitiplication**<br>
  arr: *ndarray* --- input array to convert<br>
  minimum: *float* --- minimum magnitude allowed above or below 0 (prevents multiplication)<br>
  default_positive: *bool* --- used to set any 0 in the input array to ±minimum<br>
  returns: *ndarray*<br>
---

### IO
- `def find_file(path, pattern, allow_multiple=False)`<br>
  **Returns a file from a directory with a specified glob pattern**<br>
  path: *str | Path* --- directory to search for the target file<br>
  pattern: *str* --- glob pattern with which to search<br>
  allow_multiple: *bool* --- whether to return an error if mulple files are found with the pattern<br>
  returns: *Path | list\[Path]*<br>

- `def load_file(path, pattern, allow_pickle=False)`<br>
  **Finds and loads data from a single .npy, .tif, .h5, or .hdf5 file from a directory with a specified glob pattern**<br>
  path: *str | Path* --- directory to search for the target file<br>
  pattern: *str* --- glob pattern with which to search<br>
  allow_pickle: *bool* --- if a .npy file is found, whether to allow pickling<br>
  returns: *file data (depends on file type)*<br>

- `def check_temp(clear=False)`<br>
  **Checks if there are any temporary files in your temp directory**<br>
  clear: *bool* --- deletes any temporary files in the temp directory<br>
  returns: *None*<br>

- `def clear_temp(notify=True)`<br>
  **Clears any temporary files in the temp directory**<br>
  notify: *bool* --- whether to log that files were cleared<br>
  returns: *None*<br>

- `class Memmap(shape, dtype)`<br>
  **Streamlines handling of numpy memmap objects in the temp directory**<br>
  shape: *tuple()* --- shape of the stored array<br>
  dtype: *dtype* --- dtype of the stored array<br>

  - `def save(self, data)`<br>
    **Saves a memmap object in the temp directory**<br>
    data: *ndarray* --- numpy array to save into a memmap file<br>
    returns: *None*<br>
 
  - `def load(self, read_only=True)`<br>
    **Loads a memmap object's data**<br>
    read_only: *bool* --- whether the memmap is loaded with read or read/write permissions<br>
    returns: *numpy.Memmap*<br>

  - `def delete(self)`<br>
    **Deletes a memmap object from the temp directory**<br>
    returns: *None*<br>

- `def load_fps(path)`<br>
  **Loads the fps of from an xml file found in the specified directory**<br>
  path: *str | Path* --- directory to search for the target file<br>
  returns: *float*<br>

- `def load_resolution(path)`<br>
  **Loads the (z,y,x) resolution from a txt file found in the specified directory**<br>
  path: *str | Path* --- directory to search for the target file<br>
  returns: *tuple(float, float, float)*<br>

- `def load_metadata(path)`<br>
  **Loads the (t,z,y,x) resolution from txt and xml files found in the specified directory**<br>
  path: *str | Path* --- directory to search for the metadata<br>
  returns: *tuple(float, float, float, float)*<br>

- `def load_suite2p_data(path, mode="auto")`<br>
  **Loads the labeled volume of all ROIs, cell activity traces, time-averaged brainmap, data shape, and ops from a Suite2p-containing directory**<br>
  *Requires the existence of stat.npy, F.npy, and ops.npy files*<br>
  path: *str | Path* --- directory containing Suite2p files<br>
  mode: *str* ("auto", "raw" "percentile") --- method of ΔF/F calculation<br>
  returns:<br> 
  - *ndarray (ndim of 3)* --- labeled volume (z,y,x)<br>
  - *ndarray (ndim of 2)* --- cell traces (c,t)<br>
  - *ndarray (ndim of 3)* --- brainmap (z,y,x)<br>
  - *tuple(int, int, int, int, int)* --- data shape (Lc,Lt,Lz,Ly,Lx)<br>
  - *Suite2p Ops*<br>

- `def load_voluseg_data(path, mode="auto")`<br>
  **Loads the labeled volume of all ROIs, cell activity traces, time-averaged brainmap, and data shape from a VoluSeg-containing directory**<br>
  *Requires the existence of volume0.hdf5 and cells0_clean.hdf5 files*<br>
  path: *str | Path* --- directory containing VoluSeg files<br>
  mode: *str* ("auto", "raw", "percentile", "voluseg") --- method of ΔF/F calculation<br>
  returns:<br> 
  - *ndarray (ndim of 3)* --- labeled volume (z,y,x)<br>
  - *ndarray (ndim of 2)* --- ΔF/F cell traces (c,t)<br>
  - *ndarray (ndim of 3)* --- brainmap (z,y,x)<br>
  - *tuple(int, int, int, int, int)* --- data shape (Lc,Lt,Lz,Ly,Lx)<br>

- `def load_combined_data(path, file="segdata.h5")`<br>
  **Loads the labeled volume of all ROIs, ROI activity traces, time-averaged brainmap, and data shape from a Combined-Segmentation-containing directory**<br>
  *Requires the existence of a combined_segdata.h5 file*<br>
  path: *str | Path* --- directory containing combined file<br>
  file: *str* --- name of combined file<br>
  returns:<br> 
  - *ndarray (ndim of 3)* --- labeled volume (z,y,x)<br>
  - *ndarray (ndim of 2)* --- ROI traces (r,t)<br>
  - *ndarray (ndim of 3)* --- brainmap (z,y,x)<br>
  - *ndarray ([int, int, int, int, int\])* --- data shape (Lr,Lc,Lt,Lz,Ly,Lx)<br>
  - *ndarray ([float, float, float, float\])* --- metadata (Rt,Rz,Ry,Rx)<br>
  - *ndarray (ndim of 1)* --- Ids of Suite2p-identified ROIs
  - *ndarray (ndim of 1)* --- Ids of VoluSeg-identified ROIs

- `def build_trials(drift)`<br>
  **Builds a trials-by-timepoints (Ln,Ltt) array of timepoint indices**<br>
  drift: *ndarray (ndim of 1)* --- drift time series, used to distinguish trials<br>
  min_length: *int* --- length cutoff to distinguish go period from pulses<br>
  returns: *ndarray (ndim of 2)*
---

### Form<br>
- `def form_volume(img, shape)`<br>
  **Forms a volumetric array from a tiled image, using the specified target shape**<br>
  img: *ndarray (ndim of 2)* --- original tiled image<br>
  shape: *tuple(int, int, int)* --- shape of target volume (z,y,x)<br>
  returns: *ndarray (ndim of 3)*<br>

- `def form_tiles(vol, shape)`<br>
  **Forms a tiled image from a volume, using the specified target shape**<br>
  vol: *ndarray (ndim of 3)* --- original volume<br>
  shape: *tuple(int, int)* --- shape of target tiled image<br>
  returns: *ndarray (ndim of 2)*<br>

- `def label_rois(stat, shape)`<br>
  **Labels all ROIs in a volume using a stat.npy file**<br>
  stat: *list\[dict]* --- list of pickled and sparse-labeled cells<br>
  shape: *tuple(int, int, int)* --- shape of the target volume<br>
  returns: *ndarray (ndim of 3)*<br>

- `def adjust_rois(arr)`<br>
  **Removes missing labels in a labeled array by shifting ROIs**<br>
  arr: *ndarray* --- labeled array<br>
  returns:<br>
  - *ndarray* --- relabled array<br>
  - *ndarray (ndim of 1)* --- array of indices of the original unique labels<br>
  - *list\[int]* --- list of missing indices from the original array<br>

- `def remove_rois(arr, rois, keep=False)`<br>
  **Removes the specified labels from a labeled array, leaving missing labels**<br>
  arr: *ndarray* --- labeled array<br>
  rois: *list\[int] | ndarray (ndim of 1)* --- selected labels<br>
  keep: *bool* --- if True, remove whatever rois are not selected, and if False, remove whatever rois are selected<br>
  returns: *ndarray*<br>

- `def weight_rois(rois, weights)`<br>
  **Substitutes a labeled array with each label's corresponding weight**<br>
  rois: *ndarray* --- labeled array<br>
  weights: *list\[float] | ndarray (ndim of 1)* --- weights corresponding to each label<br>
  returns: *ndarray*<br>
---

### Stats<br>
- `def pearson_corr(x, y)`<br>
  **Calculates the Pearson Correlation and its p-value for two variables**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  returns: *tuple(float, float)* --- ρ and p-value<br>

- `def spearman_corr(x, y)`<br>
  **Calculates the Spearman Rank Correlation and its p-value for two variables**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  returns: *tuple(float, float)* --- ρ and p-value<br>

- `def phi_coef(x, y)`<br>
  **Calculates the Phi Coefficient and its p-value for two binary variables**<br>
  x: *ndarray (ndim of 1, bool)*<br>
  y: *ndarray (ndim of 1, bool)*<br>
  returns: *tuple(float, float)* --- φ and p-value<br>

- `def quantile_bins(x, n_bins=NUM_BINS)`<br>
  **Calculates the bin edges for a specified number of quantile bins**<br>
  x: *ndarray*<br>
  n_bins: *int*<br>
  returns: *ndarray (ndim of 1)*<br>
---

### Info<br>
- `def hist_H(x)`<br>
  **Calculates the Shannon Entropy of a binned variable from a list of bin counts**<br>
  x: *ndarray (ndim of 1)*<br>
  returns: *float*<br>

- `def KL_H(x, k)`<br>
  **Calculates the Shannon Entropy of a continuous variable using the Kozachenko-Leonenko estimator**<br>
  x: *ndarray (ndim of 1)* --- input array<br>
  k: *int* --- number of nearest neighbors<br>
  returns: *float*<br>

- `def disc_MI(x, y, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE)`<br>
  **Calculates the Mutual Information of two discrete variables**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  normalized: *bool*<br>
  n_bins: *int | list\[int]* --- number of bins with which to discretize x and y<br>
  bin_type: *str = 'fixed' | 'quantile'* --- type of bins with which to discretize x and y<br>
  returns: *float*<br>

- `def disc_CMI(x, y, z, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE)`<br>
  **Calculates the Conditional Mutual Information of two discrete variables conditioned on a third variable**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  z: *ndarray (ndim of 1)*<br>
  normalized: *bool*<br>
  n_bins: *int | list\[int]* --- number of bins with which to discretize x, y, and z<br>
  bin_type: *str = 'fixed' | 'quantile'* --- type of bins with which to discretize x, y, and z<br>
  returns: *float*<br>

- `def disc_II(x, y, z, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE)`<br>
  **Calculates the Interaction Information of three discrete variables**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  z: *ndarray (ndim of 1)*<br>
  normalized: *bool*<br>
  n_bins: *int | list\[int]* --- number of bins with which to discretize x, y, and z<br>
  bin_type: *str = 'fixed' | 'quantile'* --- type of bins with which to discretize x, y, and z<br>
  returns: *float*<br>

- `def KSG_MI(x, y, k=NUM_BINS, normalized=False)`<br>
  **Calculates the Mutual Information of two continuous variables using the Kraskov-Stoegbauer-Grassberger estimator**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  k: *int*<br>
  normalized: *bool*<br>
  returns: *float*<br>

- `def KSG_CMI(x, y, z, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE)`<br>
  **Calculates the Conditional Mutual Information of two continuous variables conditioned on a third variable using the Kraskov-Stoegbauer-Grassberger estimator**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  z: *ndarray (ndim of 1)*<br>
  normalized: *bool*<br>
  n_bins: *int | list\[int]* --- number of bins with which to discretize x, y, and z<br>
  bin_type: *str = 'fixed' | 'quantile'* --- type of bins with which to discretize x, y, and z<br>
  returns: *float*<br>

- `def KSG_II(x, y, z, normalized=False, n_bins=NUM_BINS, bin_type=BIN_TYPE)`<br>
  **Calculates the Interaction Information of three discrete variables using the Kraskov-Stoegbauer-Grassberger estimator**<br>
  x: *ndarray (ndim of 1)*<br>
  y: *ndarray (ndim of 1)*<br>
  z: *ndarray (ndim of 1)*<br>
  normalized: *bool*<br>
  n_bins: *int | list\[int]* --- number of bins with which to discretize x, y, and z<br>
  bin_type: *str = 'fixed' | 'quantile'* --- type of bins with which to discretize x, y, and z<br>
  returns: *float*<br>
---


## License
MIT
