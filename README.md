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
- utils
  - Contains functions for a wide range of uses
- io
  > Contains functions for I/O operations and streamlined file use
> form
  - Contains functions for manipulating labeled and non-labeled arrays
> stats
  > Contains functions for statistical tests and measures
- info
    Contains functions for information theoretic tests and measures
- draw
    Contains functions for data visualization
- config
    Contains methods for updating, resetting, loading, and saving parameter settings for the module's default behaviors and arguments

Each submodule can be used in a program directly or through imports:
```python
# Method 1
import interlinked
print(interlinked.config.defaults())

# Method 2
from interlinked import config
print(config.defaults())
```

All submodules and their functions/methods are listed below[^*]:
[^*]: Only functions built for typical users are shown below. Additional functions not shown here are considered peripheral and should only be used after reading the source code.
### Utils 
- `def digitize(x, n, dtype=np.int32)` 
  - **Digitizes an input array into discretized values from a given number of fixed-width bins**
    - *x: ndarray (ndim of 1)* --- input array to digitize
    - *n: int* --- desired range of the output array
    - *dtype: dtype* --- target dtype of output array
    
- `def dff(raw, downsample=1, percentile=20, window=300)`
  > **Calculates the ΔF/F of a calcium trace using a percentile filter and a sliding window**
  > *raw: ndarray (ndim of 1)* --- input array for which to calculate ΔF/F
  > *downsample: int* --- downsampling factor (setting to 1 prevents downsampling)
  > *percentile: float* --- percentile with which to calculate the baseline of the time series
  > *window: int* --- sliding window size with which to calculate the baseline of the time series

`def divisor(arr, minimum=1, default_positive=True)`
  > **Converts an input array into a safe divisor for array division, keeping sign and preventing unintentional mulitiplication**
  > *arr: ndarray* --- input array to convert
  > *minimum: float* --- minimum magnitude allowed above or below 0 (prevents multiplication)
  > *default_positive: bool* --- used to set any 0 in the input array to ±minimum

### IO
   `def find_file(path, pattern, allow_multiple=False)`
        **Returns a file from a directory with a specified glob pattern**
        *path: str or Path* --- directory to search for the target file
        *pattern: str* --- glob pattern with which to search
        *allow_multiple: bool* --- whether to return an error if mulple files are found with the pattern

   `def load_file(path, pattern, allow_pickle=False)`
        **Finds and loads data from a single .npy, .tif, .h5, or .hdf5 file from a directory with a specified glob pattern**
        *path: str or Path* --- directory to search for the target file
        *pattern: str* --- glob pattern with which to search
        *allow_pickle: bool* --- if a .npy file is found, whether to allow pickling

   `def check_temp(clear=False)`
        **Checks if there are any temporary files in your temp directory**
        *clear: bool* --- deletes any temporary files in the temp directory

   `def clear_temp(notify=True)`
        **Clears any temporary files in the temp directory**
        *notify: bool* --- whether to log that files were cleared

   `class Memmap`
        **Streamlines handling of numpy memmap objects**
        `def save(self, data)`
            **Saves a memmap object in the temp directory**
            *data: ndarray* --- numpy array to save into a memmap file

        `def load(self, read_only=True)`
            **temp**




## License
MIT
