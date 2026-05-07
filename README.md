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
---

### Utils 
- `def digitize(x, n, dtype=np.int32)`<br>
  **Digitizes an input array into discretized values from a given number of fixed-width bins**<br>
  x: *ndarray (ndim of 1)* --- input array to digitize<br>
  n: *int* --- desired range of the output array<br>
  dtype: *dtype* --- target dtype of output array<br>
  returns: *ndarray (ndim of 1)*<br>
    
- `def dff(raw, downsample=1, percentile=20, window=300)`<br>
  **Calculates the ΔF/F of a calcium trace using a percentile filter and a sliding window**<br>
  raw: *ndarray (ndim of 1)* --- input array for which to calculate ΔF/F<br>
  downsample: *int* --- downsampling factor (setting to 1 prevents downsampling)<br>
  percentile: *float* --- percentile with which to calculate the baseline of the time series<br>
  window: *int* --- sliding window size with which to calculate the baseline of the time series<br>
  returns: *ndarray (ndim of 1)*<br>

- `def divisor(arr, minimum=1, default_positive=True)`<br>
  **Converts an input array into a safe divisor for array division, keeping sign and preventing unintentional mulitiplication**<br>
  arr: *ndarray* --- input array to convert<br>
  minimum: *float* --- minimum magnitude allowed above or below 0 (prevents multiplication)<br>
  default_positive: *bool* --- used to set any 0 in the input array to ±minimum<br>
  returns: *ndarray*<br>


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

- `class Memmap`<br>
  **Streamlines handling of numpy memmap objects**<br>
  - `def save(self, data)`<br>
    **Saves a memmap object in the temp directory**<br>
    data: *ndarray* --- numpy array to save into a memmap file<br>
    returns: *None*<br>
 
  - `def load(self, read_only=True)`<br>
    **Loads a memmap object's data**<br>
    read_only: *bool* --- whether the memmap is loaded with read or read/write permissions<br>
    returns: *numpy memmap*<br>

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






## License
MIT
