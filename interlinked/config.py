from pathlib import Path

##################################################
#               Default parameters               # 
##################################################

# Utils parameters
LOGGING_FORMAT: str = '%(asctime)s [%(levelname)s] - %(message)s'

# IO parameters
ZBRAIN_PATH:      Path = Path('...')
TEMP_DIRECTORY:   Path = Path('...')
TEMP_FILE_PREFIX: str  = '__temp__'
TEMP_FILE_SUFFIX: str  = '.dat'
CLEAR_TEMP:       bool = True

# Parallel parameters
BATCH_SIZE:  int = 1000
NUM_WORKERS: int = 100

# Info parameters
NUM_BINS: int = 5
NUM_KNN:  int = 10

# Stats parameters
ALPHA1:          float = 0.05
ALPHA2:          float = 0.01
ALPHA3:          float = 0.001
ALPHA4:          float = 0.0001
NUM_MONTE_CARLO: int = 1_000_000

# Structure parameters


# Draw parameters


# Graph parameters
RADIUS:    int = 12
NORMALIZE: bool = False
CLIP_EDGES: bool = False

