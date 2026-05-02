import os
import sys
import logging
import numpy as np
import scipy as sp
import tifffile as tiff
import interlinked as lnk
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

logging.basicConfig(level=logging.INFO, format=lnk.config.LOGGING_FORMAT)
log = logging.getLogger(__name__)


#=================================================================================================================#

pdir = Path('/home/andrej/labwork/data/260312_f1_ND3')

bdir = pdir / 'behDir'
adir = pdir / 'analysis'
gdir = pdir / 'graphs'

#=================================================================================================================#

CLEAR_TEMP = lnk.config.CLEAR_TEMP

    
#=================================================================================================================#

if __name__ == '__main__':
    lnk.io.check_temp(clear=CLEAR_TEMP)
    log.info('Analyzing dataset: %s', pdir)
    assert pdir.exists()

    rois, trcs, bmap, shape = lnk.io.load_voluseg_data(pdir)
    Lc, Lt, Lz, Ly, Lx = shape
    log.info('Data loaded (Lc, Lt, Lz, Ly, Lx): %s', shape)
    Rz, Ry, Rx, Rt = lnk.io.load_metadata(pdir)
    log.info('Metadata loaded (Rt, Rz, Ry, Rx): (%.3f, %.3f, %.3f, %.3f)', Rt, Rz, Ry, Rx)

    drft = lnk.io.load_file(bdir, 'drift*')
    trials, move_mask, wait_mask, puls_mask = lnk.io.build_trials(drft)
    Ln, Ltt = trials.shape
    log.info('Trials metadata loaded (Ln, Ltt): (%s, %s)', Ln, Ltt)

    bhvr = lnk.io.load_file(bdir, 'behavior*')
    gain = lnk.io.load_file(bdir, 'gain*')
    vlct = lnk.io.load_file(bdir, 'velocity*')
    swim = lnk.io.load_file(bdir, 'swimbout*')
    log.info('Behavior data loaded')

    #-----------------------------------------------------------------------------------#

    #...work...

