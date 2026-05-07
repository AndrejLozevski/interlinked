import os
import pickle
import logging
import interlinked as lnk

from pathlib import Path
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)


#--| Constants |------------------------------------------------------------------------#

TEMP_DIR:    Path = Path("/tmp/interlinked")
TEMP_PREFIX: str  = "__temp__"
TEMP_SUFFIX: str  = ".dat"
CONFIG_FILE: Path = TEMP_DIR / "config.pkl"


#--| Defaults |------------------------------------------------------------------------#

# Default parameters class
@dataclass
class Defaults:
    # Temporary directory parameters
    TEMP_DIR:    Path = Path("/tmp/interlinked")
    TEMP_PREFIX: str  = "__temp__"
    TEMP_SUFFIX: str  = ".dat"
    CLEAR_TEMP:   bool = False

    # Parallel parameters
    BATCH_SIZE:  int = 1000
    NUM_WORKERS: int = 100

    # Info parameters
    NUM_BINS: int = 5
    NUM_KNNS: int = 8
    NUM_ITER: int = 1_000_000

    # Stats parameters
    ALPHA1: float = 0.05
    ALPHA2: float = 0.01
    ALPHA3: float = 0.001
    ALPHA4: float = 0.0001

    # Graph parameters
    RADIUS:     int  = 10
    MIN_SIZE:   int  = 5
    NORMALIZE:  bool = False
    CLIP_EDGES: bool = False


#--| Config |---------------------------------------------------------------------------#

# Lists all parameters from a Config instance
def _list_params(inst):
    params = {}
    for p in vars(inst):
        if p.startswith("__") or callable(p):
            continue
        params[p] = getattr(inst, p)
    return params

# Config settings class
class Config:
    def __init__(self):
        self.reset()
        return

    def defaults(self):
        log.info("Config defaults: %s.", asdict(defaults))
        return

    def list(self):
        params = _list_params(self)
        log.info("Config paramters: %s.", params)
        return

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(defaults, key):
                new_type = type(value)
                old_type = type(getattr(defaults, key))
                if isinstance(new_type, old_type):
                    setattr(self, key, value)
                    continue
                try:
                    value = old_type(value)
                    setattr(self, key, value)
                    continue
                except:
                    lnk.meta.Exception("Parameter %s is of type '%s'. Cannot assign type '%s'.", key, old_type, new_type, error=TypeError)
                    continue
            else:
                setattr(self, key, value)
        return

    def reset(self):
        params = _list_params(self)
        for key, value in params.items():
            delattr(self, key)
        self.configure(**asdict(defaults))
        return

    def save(self):
        with open(CONFIG_FILE, 'wb') as file:
            pickle.dump(self, file)
        return

    def load(self):
        self.reset()
        if not CONFIG_FILE.exists():
            lnk.meta.Exception("No saved config file found in temp directory. Loading defaults.")
        else:
            with open(CONFIG_FILE, 'rb') as file:
                old = pickle.load(file)
                params = _list_params(old)
                self.configure(**params)
        return
                    

#--| Initialization |-------------------------------------------------------------------#

os.makedirs(TEMP_DIR, exist_ok=True)
defaults = Defaults()
config   = Config()
