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
from sklearn.cluster import KMeans

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

    vari = np.percentile(trcs, 99, axis=1) - np.percentile(trcs, 1, axis=1)
    keep = np.argwhere(vari >= 1.0)[:,0]
    rois = lab.structure.remove_rois(rois, keep, keep=True)
    rois, _, _ = lab.structure.adjust_rois(rois)
    trcs = trcs[keep,:]
    Lc = len(keep)
    assert Lc == len(keep) == len(np.unique(rois))-1 == rois.max()+1 == trcs.shape[0]
    log.info('Removed invariant ROIs. Lc: %s', Lc)
    del vari, keep

    def build_mask(coords, shape=(Lz,Ly,Lx)):
        cZ, cY, cX = coords
        mask = np.zeros(shape, bool)
        mask[cZ[0]:cZ[1], cY[0]:cY[1], cX[0]:cX[1]] = True
        return mask

    apost = build_mask([( 0, 8),(450,650),( 200, 325)])
    slomo = build_mask([( 0,15),(375,750),( 400, 800)])
    olive = build_mask([(21,30),(425,725),( 325, 575)])
    vagus = build_mask([(24,44),(175,300),( 400, 575)]) |  build_mask([(24,44),(825,950),( 400, 575)])
    ptect = build_mask([(26,31),(400,750),(1250,1425)]) |  build_mask([(24,27),(425,700),(1400,1650)])
    cereb = build_mask([( 6,17),(150,950),( 950,1150)]) & ~build_mask([(18,44),(400,700),( 950,1100)])
    spinl = build_mask([( 0,21),(450,650),(   0, 350)]) & ~apost
    hypot = build_mask([(33,44),(150,950),( 950,1300)])

    rgns = np.zeros(bmap.shape, np.uint8)
    rgns[apost] = 1
    rgns[slomo] = 2
    rgns[olive] = 3
    rgns[vagus] = 4
    rgns[ptect] = 5
    rgns[cereb] = 6
    rgns[spinl] = 7
    rgns[hypot] = 8

    nmap = bmap.copy()
    nmap[rgns == 0] = nmap.min()

    tiff.imwrite(pdir / 'bmap.tif', bmap.astype(np.float32), **lab.io.TIF3D)
    tiff.imwrite(pdir / 'rgns.tif', rgns, **lab.io.TIF3D)
    tiff.imwrite(pdir / 'nmap.tif', nmap.astype(np.float32), **lab.io.TIF3D)
    log.info('Regions identified')

    c_apost = [c for c in np.unique(rois[apost]) if c != -1]
    c_slomo = [c for c in np.unique(rois[slomo]) if c != -1]
    c_olive = [c for c in np.unique(rois[olive]) if c != -1]
    c_vagus = [c for c in np.unique(rois[vagus]) if c != -1]
    c_ptect = [c for c in np.unique(rois[ptect]) if c != -1]
    c_cereb = [c for c in np.unique(rois[cereb]) if c != -1]
    c_spinl = [c for c in np.unique(rois[spinl]) if c != -1]
    c_hypot = [c for c in np.unique(rois[hypot]) if c != -1]
    log.info('Cells identified')

    #k = 4
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_apost,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_apost, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_apost.tif', fmap, **lab.io.TIF3D)

    #k = 4
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_slomo,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_slomo, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_slomo.tif', fmap, **lab.io.TIF3D)

    #k = 7
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_olive,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_olive, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_olive.tif', fmap, **lab.io.TIF3D)
    #k_ = range(1,20)

    #k = 5
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_vagus,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_vagus, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_vagus.tif', fmap, **lab.io.TIF3D)

    #k = 4
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_ptect,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_ptect, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_ptect.tif', fmap, **lab.io.TIF3D)

    #k = 5
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_cereb,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_cereb, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_cereb.tif', fmap, **lab.io.TIF3D)

    #k = 3
    #kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #kmns.fit(trcs[c_spinl,:])
    #lbls = kmns.labels_
    #fmap = -1 * np.ones(bmap.shape, np.uint8)
    #for c, l in zip(c_spinl, lbls):
    #    fmap[rois == c] = l
    #tiff.imwrite(pdir / 'fmap_spinl.tif', fmap, **lab.io.TIF3D)

    k = 3
    kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    kmns.fit(trcs[c_hypot,:])
    lbls = kmns.labels_
    fmap = -1 * np.ones(bmap.shape, np.uint8)
    for c, l in zip(c_hypot, lbls):
        fmap[rois == c] = l
    tiff.imwrite(pdir / 'fmap_hypot.tif', fmap, **lab.io.TIF3D)

    #k_ = range(1,20)
    #inertias = []
    #for k in k_:
    #    kmns = KMeans(n_clusters=k, random_state=1, n_init='auto')
    #    kmns.fit(trcs[c_hypot,:])
    #    inertias.append(kmns.inertia_)
    #plt.figure()
    #plt.plot(k_, inertias, 'o-')
    #plt.savefig(pdir / 'elbow.svg')
    #plt.close()
























