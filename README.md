# The Interlinked Module
This module is for neuroscientific computation and analysis with Python submodules and Rust-optimized kernels.
This code is developed by Andrej Lozevski for the En Yang Lab at UNC Chapel Hill (2026).


## Installation
This module can be installed by running:
```bash
pip install interlinked-lab
```
___

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
---

All submodules and their functions/methods are listed below[^1]:
[^1]: Only functions built for typical users are shown below. Additional functions not shown here are considered peripheral and should only be used after reading the source code.

### Config 
This submodule is a class instance, possessing the following methods and attributes:

`def defaults(self)`
- **Lists all config default parameters**
- returns: *None*

`def list(self)`
- **Lists all current config parameters**
- returns: *None*

`def configure(self, \*\*kwargs)`
- **Updates any specified config parameters**
- *Default parameters cannot be changed to a different type, but custom parameters have no restrictions*
- returns: *None*

`def reset(self)`
- **Resets config parameters to defaults**
- returns: *None*

`def save(self)`
- **Saves current config parameters in the temp directory**
- *Overwrites any pre-existing saved config parameters*
- returns: *None*

`def load(self)`
- **Loads config parameters saved in the temp directory**
- *If no saved parameters are found, throws an Exception and loads defaults*
- returns: *None*

Attributes:
- *TEMP_DIR: Path = /tmp/interlinked* --- where the temp directory is located **(DO NOT CHANGE)**
- *TEMP_PREFIX: str = "\_\_temp\_\_"* --- file prefix given to temporary files **(DO NOT CHANGE)**
- *TEMP_SUFFIX: str = ".dat"* --- file suffix given to temporary files **(DO NOT CHANGE)**
- *CLEAR_TEMP: bool = False* --- *see io.check_temp() below*

- *BATCH_SIZE: int = 1000* --- size of batches for parallelism
- *NUM_WORKERS: int = 8* --- number of workers for parallelism

- *NUM_BINS: int = 5* --- number of bins for binning and digitization
- *NUM_KNNS: int = 8* --- number of nearest neighbors for KNN operations
- *NUM_ITER: int = 1_000_000* --- number of iterations for monte carlo simulation and bootstrap resampling operations

- *ALPHA1: float = 0.05* --- α₁ for statistical tests
- *ALPHA2: float = 0.01* --- α₂ for statistical tests
- *ALPHA3: float = 0.001* --- α₃ for statistical tests
- *ALPHA4: float = 0.0001* --- α₄ for statistical tests

- *RADIUS: int* --- radius used for graph theory and nearest neighbors operations
- *MIN_SIZE: int* --- minimum node count for graph theory operations
***

### Utils 
- `def digitize(x, n, dtype=np.int32)`
  &emsp;**Digitizes an input array into discretized values from a given number of fixed-width bins**
  &emsp;*x: ndarray (ndim of 1)* --- input array to digitize
  &emsp;*n: int* --- desired range of the output array
  &emsp;*dtype: dtype* --- target dtype of output array
  &emsp;returns: *ndarray (ndim of 1)*
    
<pre>
- `def dff(raw, downsample=1, percentile=20, window=300)`  
  **Calculates the ΔF/F of a calcium trace using a percentile filter and a sliding window**  
  *raw: ndarray (ndim of 1)* --- input array for which to calculate ΔF/F  
  *downsample: int* --- downsampling factor (setting to 1 prevents downsampling)  
  *percentile: float* --- percentile with which to calculate the baseline of the time series  
  *window: int* --- sliding window size with which to calculate the baseline of the time series  
  returns: *ndarray (ndim of 1)*  
</pre>

- `def divisor(arr, minimum=1, default_positive=True)`
  &ensp;**Converts an input array into a safe divisor for array division, keeping sign and preventing unintentional mulitiplication**
  &ensp;*arr: ndarray* --- input array to convert
  &ensp;*minimum: float* --- minimum magnitude allowed above or below 0 (prevents multiplication)
  &ensp;*default_positive: bool* --- used to set any 0 in the input array to ±minimum
  &ensp;returns: *ndarray*


### IO
`def find_file(path, pattern, allow_multiple=False)`
- **Returns a file from a directory with a specified glob pattern**
- *path: str | Path* --- directory to search for the target file
- *pattern: str* --- glob pattern with which to search
- *allow_multiple: bool* --- whether to return an error if mulple files are found with the pattern
- returns: *Path*

`def load_file(path, pattern, allow_pickle=False)`
- **Finds and loads data from a single .npy, .tif, .h5, or .hdf5 file from a directory with a specified glob pattern**
- *path: str | Path* --- directory to search for the target file
- *pattern: str* --- glob pattern with which to search
- *allow_pickle: bool* --- if a .npy file is found, whether to allow pickling
- returns: *file data (depends on file type)*

`def check_temp(clear=False)`
- **Checks if there are any temporary files in your temp directory**
- *clear: bool* --- deletes any temporary files in the temp directory
- returns: *None*

`def clear_temp(notify=True)`
- **Clears any temporary files in the temp directory**
- *notify: bool* --- whether to log that files were cleared
- returns: *None*

`class Memmap`
- **Streamlines handling of numpy memmap objects**
- `def save(self, data)`
  - **Saves a memmap object in the temp directory**
  - *data: ndarray* --- numpy array to save into a memmap file
  - returns: *None*
 
- `def load(self, read_only=True)`
  - **Loads a memmap object's data**
  - *read_only: bool* --- whether the memmap is loaded as *'r'* or *'r+'*
  - returns: *numpy memmap*

- `def delete(self)`
  - **Deletes a memmap object from the temp directory**
  - returns: *None*

`def load_fps(path)`
- **Loads the fps of from an xml file found in the specified directory**
- *path: str | Path* --- directory to search for the target file
- returns: *float*

`def load_resolution(path)`
- **Loads the (z,y,x) resolution from a txt file found in the specified directory**
- *path: str | Path* --- directory to search for the target file
- returns: *tuple(float, float, float)*

`def load_metadata(path)`
- **Loads the (t,z,y,x) resolution from txt and xml files found in the specified directory**
- *path: str | Path* --- directory to search for the metadata
- returns: *tuple(float, float, float, float)*






## License
MIT
