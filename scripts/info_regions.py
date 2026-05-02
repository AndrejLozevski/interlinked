import os
import sys
import logging
import numpy as np
import scipy as sp
import networkx as nx
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
title = 'r10a1m5i_clean'

bdir = pdir / 'behDir'
adir = pdir / 'analysis'
gdir = pdir / 'graphs'


#=================================================================================================================#

CLEAR_TEMP = lnk.config.CLEAR_TEMP

RADIUS   = lnk.config.RADIUS
MIN_SIZE = lnk.config.MIN_SIZE


#=================================================================================================================#

def draw_frame(i, bkgd, groups, shape):
    Lc, Lt, Lz, Ly, Lx = shape
    weights = np.memmap(gdir / 'temp_weights.dat', mode='r', dtype=np.float32, shape=(Lz,Ly,Lx))
    traces = np.memmap(gdir / 'temp_traces.dat', mode='r', dtype=np.float32, shape=(Lc,Lt))
    L_ = min(Lt, Lx+Lz)

    temp = (weights == i).astype(np.uint8)
    _trns = temp.max(axis=0).astype(bool)
    _sgtl = temp.max(axis=1).astype(bool)
    _crnl = temp.max(axis=2).T.astype(bool)
     
    bkgd[:Ly,:Lx][_trns]  = 255
    bkgd[:Ly,-Lz:][_crnl] = 255
    bkgd[-(Lz+121):-121,:Lx][_sgtl] = 255

    count = np.sum(groups == i)
    text = np.zeros((Lz, Lz), np.uint8)
    cv2.putText(text, str(count), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 5, cv2.LINE_AA)
    bkgd[-(Lz+121):-121,-Lz:] = text

    _traces = traces[groups==i,:][:,:L_]
    _traces = (_traces - _traces.min()) / (_traces.max() - _traces.min())
    _traces *= 100
    assert _traces.shape[0] == count and _traces.shape[1] == L_

    board = 255 * np.ones((100,L_), np.uint8)

    mean = np.rint(np.mean(_traces, axis=0)).astype(np.uint8)
    mean = mean - mean.mean()
    stdv = np.rint(np.std(_traces,  axis=0)).astype(np.uint8)
    for t in range(L_//3):
        v = mean[t]
        s = stdv[t]
        center = int(v) + board.shape[0]//2
        board[center:center+3,t*3:3+t*3] = 0
    bkgd[-100:,:L_] = board
    return i, bkgd
    
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
    swim = lnk.io.load_file(bdir, 'swim*bout*')
    log.info('Behavior data loaded')

    lnk.form.validate_rois(rois, Lc)

    vari = np.percentile(trcs, 99, axis=1) - np.percentile(trcs, 1, axis=1)
    keep = np.argwhere(vari >= 1.0)[:,0]
    rois = lnk.form.remove_rois(rois, keep, keep=True)
    rois, _, _ = lnk.form.adjust_rois(rois)
    trcs = trcs[keep,:]
    Lc = len(keep)
    lnk.form.validate_rois(rois, Lc)
    log.info('Removed invariant ROIs. Lc: %s', Lc)
   
    count = 1_000
    rand1 = np.random.randint(0, Lc, size=count)
    rand2 = np.random.randint(0, Lc, size=count)
    vals = np.zeros(len(rand1), np.float32)
    for j in tqdm(range(count)):
        vals[j] = lnk.info.KSG_MI(trcs[rand1[j],:], trcs[rand2[j],:], 10)
    min_MI = np.percentile(vals, 95)
    log.info('min MI: %.3f', min_MI)
    raise Exception

    mtrx = lab.graph.build_rgraph(lab.info.ksg_MI, rois=rois, traces=trcs, resolution=(Rz,Ry,Rx), alpha=lab.config.ALPHA1, statistic=min_MI, k=np.int32(10), radius=params['rads'])
    sp.sparse.save_npz(adir / f'region_matrix_{title}.npz', mtrx)
    mtrx = sp.sparse.load_npz(adir / f'region_matrix_{title}.npz')
    log.info('Regions graph built')

    weights = lab.draw.draw_graph_heatmap(mtrx, rois)
    tiff.imwrite(adir / f'region_heatmap_{title}.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / f'region_heatmap_{title}.pdf', resolution=(Rz,Ry,Rx))

    groups, weights = lab.graph.label_graph(mtrx, rois, min_size=params['size'])
    tiff.imwrite(adir / f'region_labels_{title}.tif', weights.astype(np.float32), **lab.io.TIF3D)
    np.save(adir / f'region_groups_{title}.npy', groups.astype(np.float32))

    #groups = np.load(adir / 'region_groups_r20a3m5.npy')
    log.info('Regions identified: %s', len(np.unique(groups)) - 1)
    log.info('Grouped cells: %s', len(np.where(groups > 0)[0]))
    log.info('Ungrouped cells: %s', len(np.where(groups == 0)[0]))
    log.info('Total cells: %s', groups.shape[0])
    #log.info('Group 1 cell count: %s', len(np.where(groups == 1)[0]))

        #test = np.where(groups == 1)[0]
        #_trcs = trcs[test,:]
        #_trcs = _trcs[:,trials[frwd,:]]
        #print('>> ', trcs.shape, _trcs.shape)
        #plt.figure()
        #plt.plot(drft[trials[frwd[0],:]], color='blue')
        ##plt.plot(np.mean(trcs[test,:][:,trials[frwd,:]], axis=(0,1)), color='red')
        #plt.plot(_trcs.mean(axis=(0,1)), color='red')
        #for i in range(10):
        #    #plt.plot(np.mean(trcs[test[i*100],:][frwd,:], axis=0), color='black', alpha=0.2)
        #    plt.plot(_trcs[i*10,:,:].mean(axis=0), color='black', alpha=0.2)
        #plt.savefig(adir / 'hahahahahah.pdf')
        #plt.close()



    #raise Exception
    
    def _draw_loc(path, regn, resolution, overlay=False):
        temp = np.zeros(Lc, np.float32)
        temp[regn] = 10
        weights = lab.draw.draw_weights(rois, temp)
        #tiff.imwrite(path.with_suffix('.tif'), weights, **lab.io.TIF3D)
        lab.draw.draw_projections(weights, bmap, path.with_suffix('.pdf'), resolution=resolution, overlay=overlay)

    def _draw_act(path, regn, title):
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(drft[trials[bkwd[0],:]], color='blue')
        for i in range(len(regn)):
            plt.plot(trcs[regn[i],:][trials[bkwd,:]].mean(axis=0), color='black', alpha=0.1)
        plt.plot(trcs[regn,:][:,trials[bkwd,:]].mean(axis=(0,1)), color='red', alpha=0.5)
        plt.ylim(-1.0, 2.5)
        plt.title(title)

        plt.subplot(3,1,2)
        plt.plot(drft[trials[nowd[0],:]], color='blue')
        for i in range(len(regn)):
            plt.plot(trcs[regn[i],:][trials[nowd,:]].mean(axis=0), color='black', alpha=0.1)
        plt.plot(trcs[regn,:][:,trials[nowd,:]].mean(axis=(0,1)), color='red', alpha=0.5)
        plt.ylim(-1.0, 2.5)

        plt.subplot(3,1,3)
        plt.plot(drft[trials[frwd[0],:]], color='blue')
        for i in range(len(regn)):
            plt.plot(trcs[regn[i],:][trials[frwd,:]].mean(axis=0), color='black', alpha=0.1)
        plt.plot(trcs[regn,:][:,trials[frwd,:]].mean(axis=(0,1)), color='red', alpha=0.5)
        plt.ylim(-1.0, 2.5)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _draw(path, regn, resolution, title):

        plt.figure()
        plt.subplot(3,2,1)


        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(drft[trials[bkwd[0],:]], color='blue')
        for i in range(len(regn)):
            plt.plot(trcs[regn[i],:][trials[bkwd,:]].mean(axis=0), color='black', alpha=0.1)
        plt.plot(trcs[regn,:][:,trials[bkwd,:]].mean(axis=(0,1)), color='red', alpha=0.5)
        plt.ylim(-1.0, 2.5)
        plt.title(title)

        plt.subplot(3,1,2)
        plt.plot(drft[trials[nowd[0],:]], color='blue')
        for i in range(len(regn)):
            plt.plot(trcs[regn[i],:][trials[nowd,:]].mean(axis=0), color='black', alpha=0.1)
        plt.plot(trcs[regn,:][:,trials[nowd,:]].mean(axis=(0,1)), color='red', alpha=0.5)
        plt.ylim(-1.0, 2.5)

        plt.subplot(3,1,3)
        plt.plot(drft[trials[frwd[0],:]], color='blue')
        for i in range(len(regn)):
            plt.plot(trcs[regn[i],:][trials[frwd,:]].mean(axis=0), color='black', alpha=0.1)
        plt.plot(trcs[regn,:][:,trials[frwd,:]].mean(axis=(0,1)), color='red', alpha=0.5)
        plt.ylim(-1.0, 2.5)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    groups = groups.astype(np.int32)
    #rdir = adir / 'regns_r20a3m5'
    #rdir.mkdir(exist_ok=True)

    #for i in tqdm(range(1,groups.max()+1), desc='labeling'):
    #    regn = np.where(groups == i)[0]
    #    #_draw_loc(rdir / f'regn_{i}_loc.pdf', regn, resolution=(Rz,Ry,Rx))
    #    #_draw_loc(rdir / f'regn_{i}_locO.pdf', regn, resolution=(Rz,Ry,Rx), overlay=True)
    #    #_draw_act(rdir / f'regn_{i}_act.pdf', regn, f'Regn {i} ({len(regn)} rois)')
    #    lab.draw.draw_region(rdir / f'regn_{i}.pdf', brainmap=bmap, rois=rois, region=regn, region_id=i, traces=trcs, drift=drft, trials=trials, frwd=frwd, nowd=nowd, bkwd=bkwd, resolution=(Rz,Ry,Rx))

    _weights_ = weights.copy()
    weights = np.repeat(weights, Rz//Rx, axis=0)
    norm = np.repeat(bmap, Rz//Rx, axis=0).astype(np.float32)
    norm = (norm-norm.min())/(norm.max()-norm.min())
    norm *= 200
    norm = np.rint(norm).astype(np.uint8)

    Lz, Ly, Lx = norm.shape
    _trns = norm.max(axis=0)
    _sgtl = norm.max(axis=1)
    _crnl = norm.max(axis=2).T
    
    lbls = np.zeros((groups.max(), Ly+Lz+121, Lx+Lz), np.uint8)
    lbls[:,:Ly,:Lx]  = _trns[None,:,:]
    lbls[:,:Ly,-Lz:] = _crnl[None,:,:]
    lbls[:,-(Lz+121):-121,:Lx] = _sgtl[None,:,:]
    lbls[:,-121:,:] = 255
    
    for t in tqdm(range(min(Lt, Lx+Lz)//3), desc='drawing drift'):
        if drft[t] == 0.0:
            lbls[:,-110:,t*3:3+t*3] = 0
        elif drft[t] <= -1.0:
            lbls[:,-102:,t*3:3+t*3] = 0
        elif drft[t] >= 1.0:
            lbls[:,-118:,t*3:3+t*3] = 0

    bkgd = lbls[0,:,:].copy()

    #for i in tqdm(range(1, groups.max()+1), desc='labeling'):
    #    temp = (weights == i).astype(np.uint8)
    #    _trns = temp.max(axis=0).astype(bool)
    #    _sgtl = temp.max(axis=1).astype(bool)
    #    _crnl = temp.max(axis=2).T.astype(bool)

    #    lbls[i-1,:Ry,:Rx][_trns] = 200
    #    lbls[i-1,:Ry,-Rz:][_crnl] = 200
    #    lbls[i-1,-Rz:,:Rx][_sgtl]  = 200

    #    count = np.sum(groups == i)

    #    text = np.zeros((Rz, Rz), np.uint8)
    #    cv2.putText(text, str(count), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 3, 200, 5, cv2.LINE_AA)
    #    lbls[i-1, -Rz:, -Rz:] = text

    #tiff.imwrite(gdir / 'regions_r12a3m5_2.tif', lbls, **lab.io.TIF3D)


    _weights = np.memmap(gdir / 'temp_weights.dat', mode='w+', dtype=np.float32, shape=(Lz,Ly,Lx))[:]
    _weights[:] = weights
    _traces = np.memmap(gdir / 'temp_traces.dat', mode='w+', dtype=np.float32, shape=(Lc,Lt))[:]
    _traces[:] = trcs
    del _weights, _traces

    pfxn = partial(draw_frame, bkgd=bkgd, groups=groups, shape=(Lc,Lt,Lz,Ly,Lx))

    results = []
    contex = mp.get_context('spawn')
    num_workers = np.minimum(lab.config.NUM_WORKERS, mp.cpu_count())
    with contex.Pool(processes=num_workers) as pool:
        processes = [pool.apply_async(pfxn, (i, )) for i in range(1, groups.max() + 1)]
        for process in tqdm(processes, total=groups.max(), desc=lab.utils.format_log(desc='Drawing regions')):
            result = process.get()
            lbls[result[0]-1,:,:] = result[1]
    tiff.imwrite(gdir / f'regions_{title}.tif', lbls, **lab.io.TIF3D)
    log.info('Finished drawing regions')

    
    colr = lab.draw.draw_rois(_weights_.astype(np.int64), _weights_.shape)
    assert colr.max() < 65535 and len(np.unique(colr)) < 65535
    if colr.min() == -1:
        colr += 1
    tiff.imwrite(gdir / '___roi_colr___.tif', colr.astype(np.uint16), imagej=True, metadata={'axes':'ZCYX'})

    
    raise Exception
    log.info('Parallelizing drawings')
    pfxn = partial(lab.draw.draw_region, brainmap=bmap, rois=rois, traces=trcs, drift=drft, trials=trials, frwd=frwd, nowd=nowd, bkwd=bkwd, resolution=(Rz,Ry,Rx))

    results = []
    contex = mp.get_context('spawn')
    num_workers = np.minimum(lab.config.NUM_WORKERS, mp.cpu_count())
    with contex.Pool(processes=num_workers) as pool:
        processes = [pool.apply_async(pfxn, (rdir / f'regn_{i}.pdf', np.where(groups == i)[0], i)) for i in range(1, groups.max() + 1)]

        _ = 0
        for process in tqdm(processes, total=groups.max(), desc=lab.utils.format_log(desc='Drawing regions')):
            __ = process.get()
            _ += 1
    log.info('Finished drawing regions')
    print(_)



    raise Exception
    #mtrx = lab.graph.build_rgraph(lab.stats.pearson, rois=rois, traces=trcs, resolution=(Rz,Ry,Rx), alpha=lab.config.ALPHA3, statistic=0.875, fast=True)
    #sp.sparse.save_npz(adir / 'region_matrix.npz', mtrx)
    mtrx = sp.sparse.load_npz(adir / 'region_matrix.npz')
    log.info('Regions graph built')

    weights = lab.draw.draw_graph_heatmap(mtrx, rois)
    tiff.imwrite(adir / 'region_heatmap.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'region_heatmap.pdf')
    
    groups, weights = lab.graph.label_graph(mtrx, rois, min_size=5)
    tiff.imwrite(adir / 'region_labels.tif', weights.astype(np.float32), **lab.io.TIF3D)
    np.save(adir / 'region_groups.npy', groups.astype(np.float32))


    

    raise Exception

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], drftD[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_driftD.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], drftDi[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_driftDi.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], bhvrD[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_bhvrD.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], bhvrDi[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_bhvrDi.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], planDi, fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_planDi.pdf', pmin=75, pmax=99.9)













    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], predDi, fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_predDi.pdf', pmin=75, pmax=99.9)



    #-------------------------------------------------------------------------------------------------------------#
    raise Exception

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], drft[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_drift.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], puls, fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_pulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_pulse.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], Fpls, fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_Fpulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_Fpulse.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], Bpls, fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_Bpulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_Bpulse.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], Npls, fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_Npulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_Npulse.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], behavior[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_behavior.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_behavior.pdf', pmin=75, pmax=99.9)

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], swim[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(gdir / 'spmn_swim.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spmn_swim.pdf', pmin=75, pmax=99.9)





    #-------------------------------------------------------------------------------------------------------------#
    raise Exception

    _MIs = lab.parallel.parallelize(lab.info.discrete_MI, trcs[:,trials.ravel()], drft[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'MI_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'MI_drift.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_cMI, trcs[:,trials.ravel()], drft[trials.ravel()], bhvr[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'cMI_visual.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'cMI_visual.pdf', pmin=85, pmax=99.99)
    

    _MIs = lab.parallel.parallelize(lab.info.discrete_MI, trcs[:,trials.ravel()], bhvr[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'MI_behavior.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'MI_behavior.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_cMI, trcs[:,trials.ravel()], bhvr[trials.ravel()], drft[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'cMI_motor.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'cMI_motor.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_iMI, trcs[:,trials.ravel()], bhvr[trials.ravel()], drft[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'iMI_sensorimotor.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'iMI_sensorimotor.pdf', pmin=85, pmax=99.99)



    _MIs = lab.parallel.parallelize(lab.info.discrete_MI, trcs[:,trials.ravel()], puls, normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'MI_pulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'MI_pulse.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_cMI, trcs[:,trials.ravel()], puls, drft[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'cMI_pulse_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'cMI_pulse_drift.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_iMI, trcs[:,trials.ravel()], bhvr[trials.ravel()], puls, normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'iMI_pulse_behavior.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'iMI_pulse_behavior.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_MI, trcs[:,trials.ravel()], wait, normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'MI_waiting.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'MI_waiting.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_cMI, trcs[:,trials.ravel()], wait, puls, normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'cMI_waiting_pulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'cMI_waiting_pulse.pdf', pmin=85, pmax=99.99)


    _MIs = lab.parallel.parallelize(lab.info.discrete_iMI, trcs[:,trials.ravel()], wait, puls, normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'iMI_waiting_pulse.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'iMI_waiting_pulse.pdf', pmin=85, pmax=99.99)


    dcay = drft.astype(np.float32)
    tau = 0.5
    krnl = tau * np.arange(5)
    dcay = np.convolve(dcay, krnl, mode='full')[:Lt]
    dcay = np.clip(dcay, 0, 1.0)

    _MIs = lab.parallel.parallelize(lab.info.discrete_MI, trcs[:,trials.ravel()], dcay[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'MI_decay.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'MI_decay.pdf', pmin=85, pmax=99.99)

    dcay = np.clip(dcay, 0, 1.0)
    _MIs = lab.parallel.parallelize(lab.info.discrete_MI, trcs[:,trials.ravel()], dcay[trials.ravel()], normalized=True)
    MIs = np.zeros(Lc, np.float32)
    for c, MI in _MIs:
        MIs[c] = MI
    print(MIs.min(), MIs.mean(), MIs.max())
    weights = lab.draw.draw_weights(rois, MIs)
    #tiff.imwrite(gdir / 'MI_decay.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'MI_decay2.pdf', pmin=85, pmax=99.99)















    #-------------------------------------------------------------------------------------------------------------#
    raise Exception

    spearman = lab.parallel.parallelize(lab.stats.spearman, trcs[:,trials.ravel()], drft[trials.ravel()], fast=True)
    corrs = np.zeros((Lc, 2), np.float32)
    for c, (corr, pval) in spearman:
        corrs[c,:] = corr, pval
    weights = lab.draw.draw_weights(rois, corrs[:,0])
    tiff.imwrite(gdir / 'spearman_drift.tif', weights, **lab.io.TIF3D)
    lab.draw.draw_projections(weights, bmap, gdir / 'spearman_drift.pdf', pmin=75, pmax=99.9)

    regn1 = np.unique(rois[16:,:200,1100:1500])
    regn2 = np.unique(rois[16:,900:,1100:1500])
    regn = np.concatenate((regn1, regn2))
    regn = np.unique(regn[regn >= 0])
    regn = np.intersect1d(regn, np.where(corrs[:,0] > np.percentile(corrs[:,0], 98)))
    print(len(regn))

    temp = np.zeros(Lc, np.float32)
    for i in range(len(regn)):
        temp[regn[i]] = 1.0
    weights = lab.draw.draw_weights(rois, temp)
    lab.draw.draw_projections(weights, bmap, gdir / 'regn_spearman.pdf')
    np.save(adir / 'regn.npy', regn)


    #-------------------------------------------------------------------------------------------------------------#
    raise Exception

    _avg = trcs.mean(axis=1)
    weights = lab.draw.draw_weights(rois, _avg)
    tiff.imwrite(gdir / 'averaged_dff.tif', weights, **lab.io.TIF3D)
    #raise Exception

    swim = swim.astype(bool)
    dswim = (trcs[:,~swim].mean(axis=1) - trcs[:,swim].mean(axis=1)) / trcs[:,:].mean(axis=1)
    weights = lab.draw.draw_weights(rois, dswim)
    tiff.imwrite(gdir / 'dswim.tif', weights, **lab.io.TIF3D)

    region = rois[5:9,125:235,250:360]
    #region = rois[8:12,160:265,215:300]
    #region = rois[4:10,200:300,150:260]
    #region = rois[5:10,125:240,285:375]
    tiff.imwrite(gdir / 'region_orig.tif', region.astype(np.float32), **lab.io.TIF3D)
    region = np.unique(region[region >= 0])
    slomo = np.intersect1d(np.intersect1d(region, np.where(dswim > 0)), np.where(dswim > np.percentile(dswim, 95)))
    print(slomo.shape)

    test = np.zeros(bmap.shape, np.float32)
    test[rois >= 0] = 0
    test[np.isin(rois, slomo)] = 50
    tiff.imwrite(gdir / 'testing_slomo.tif', test, **lab.io.TIF3D)

    #tiff.imwrite(gdir / 'rois.tif', rois.astype(np.float32), **lab.io.TIF3D)
    
    slomo_mean_dff = trcs[slomo,idle_times].mean(axis=1)
    np.save(gdir / 'slomo_mean_dff.npy', slomo_mean_dff)

    print(slomo_mean_dff.shape)
    raise Exception

    #dffs = np.zeros(trcs.shape, np.float32)
    #for c in range(Lc):
    #    dffs[c,:] = lab.utils.dff(trcs[c,:])

    movie = np.zeros((100,Ly,Lx), np.float32)
    for t in tqdm(range(100)):
        movie[t,:,:] = lab.draw.draw_weights(rois, trcs[:,t])[14,:,:]
        #movie[t,:,:] = lab.draw.draw_weights(rois, dffs[:,t])[14,:,:]
    tiff.imwrite(adir / 'movie.tif', movie, **lab.io.TIF3D)
    
    #raise Exception

    dff = lab.utils.dff(bhvr)
    swim = (dff > dff.mean() + dff.std())

    plt.figure()
    #plt.plot((bhvr[:400] - bhvr[:400].mean())*3, color='black')
    #plt.plot((bhvr - bhvr.mean())*3, color='black')
    plt.plot(dff, color='black')
    #plt.plot((dff - dff.mean())*3, color='black')
    #plt.plot(test, color='black')
    plt.axhline(y=dff.mean(), color='red', alpha=1.0)
    plt.axhline(y=dff.mean() + dff.std(), color='red', alpha=0.5)
    plt.axhline(y=dff.mean() + dff.std()*2, color='red', alpha=0.25)
    plt.axhline(y=dff.mean() + dff.std()*3, color='red', alpha=0.125)
    plt.axhline(y=dff.mean() + dff.std()*1.25, color='green', alpha=0.5)
    plt.plot(drft*0.1, color='blue')
    plt.xlim(0,500)
    #plt.xlim(500,1000)
    #plt.xlim(1000,1500)
    #plt.ylim(-0.125,0.25)
    plt.savefig(pdir / '________check.png', dpi=300)
    #raise Exception

    log.info('Analyzing cells against behavior')
    #correlations = lab.parallel.parallelize(lab.stats.spearman, trcs, bhvr)
    #correlations = lab.parallel.parallelize(lab.stats.spearman, trcs, np.arange(Lt))
    #correlations = lab.parallel.parallelize(lab.stats.pearson, trcs, np.arange(Lt))
    #correlations = lab.parallel.parallelize(lab.stats.pearson, trcs, np.random.rand(Lt))
    #correlations = lab.parallel.parallelize(lab.stats.pearson, trcs, bhvr)

    dffs = (trcs[:,swim].mean(axis=1) - trcs[:,~swim].mean(axis=1)) / trcs.mean(axis=1)
    #weights = dffs.mean(axis=1)
    weights = dffs
    weights = lab.draw.draw_weights(rois, weights)
    tiff.imwrite(adir / 'temp_swim_dff.tif', weights, **lab.io.TIF3D)

    top10p_dffs = np.argsort(dffs)[:np.percentile(dffs, 10)]
    print('top10p_dffs -->', len(top10p))

    correlations = lab.parallel.parallelize(lab.stats.pearson, trcs, bhvr)
    corrs = np.zeros((Lc, 2), np.float32)
    for i, (corr, pval) in correlations:
        corrs[c,:] = corr, pval
    prsn = np.intersect1d(np.where(corrs[:,0] > 0.0), np.where(corrs[:,1] < 0.01))
    print('prsn -->', len(prsn))
    top10p_prsn = np.argsort(corrs[prsn,0])








    raise Exception

    correlations = lab.parallel.parallelize(lab.stats.pearson, ttrcs.mean(axis=1), bhvr[trials].mean(axis=0))

    corrs = np.zeros((Lc, 2), np.float32)
    for i, (corr, pval) in correlations:
        corrs[c,:] = corr, pval

    cr = corrs[:,0]
    print(cr.min(axis=0), cr.mean(axis=0), cr.std(axis=0), cr.max(axis=0))
    pv = corrs[:,1]
    print(pv.min(axis=0), pv.mean(axis=0), pv.std(axis=0), pv.max(axis=0))

    print('Lc', len(pv))
    print('sig (<0.05) cells: ', np.sum(pv < 0.05), 'not: ', np.sum(pv >= 0.05))
    print('sig (<0.01) cells: ', np.sum(pv < 0.01), 'not: ', np.sum(pv >= 0.01))

    weights = lab.draw.draw_weights(rois, cr)
    tiff.imwrite(adir / 'spearman_corr_behavior.tif', weights, **lab.io.TIF3D)

    pv += 1
    weights = lab.draw.draw_weights(rois, pv)
    weights[weights > 1.01] = 0
    weights[weights >= 1] = 1
    tiff.imwrite(adir / 'spearman_sig_behavior.tif', weights, **lab.io.TIF3D)

    raise Exception

    np.save(adir / 'spearman_behavior.npy', corrs)
    log.info('Analyzed cells against behavior')

    #corrs = np.clip(corrs, 1e-5, None)
    #print('sig (0.05) cells: ', np.sum((corrs < 0.05)[:,1]))

    #weights = lab.draw.draw_weights(rois, corrs[:,0])
    #tiff.imwrite(adir / 'spearman_corr_behavior.tif', weights, **lab.io.TIF3D)

    #weights = lab.draw.draw_weights(rois, corrs[:,1])
    ##weights[weights > 0.05] = 0
    ##weights[weights > 0] = 1
    #tiff.imwrite(adir / 'spearman_sig_behavior.tif', weights, **lab.io.TIF3D)






    
    raise Exception
    deltas = np.zeros(Ln, np.float32)
    for n in tqdm(range(Ln), desc='[PROC] comparing dffs on swim'):
        swim_mean = ttrcs[n,:,:][tswim.astype(bool)].mean()
        nswm_mean = ttrcs[n,:,:][~tswim.astype(bool)].mean()
        deltas[n] = swim_mean - nswm_mean

    weights = lab.draw.draw_weights(rois, deltas)
    tiff.imwrite(os.path.join(odir, 'dff_swim.tif'), weights, imagej=True, metadata={'axes':'ZYX'})

    indx = np.argsort(deltas)
    motor = indx[-50:]
    antimotor = indx[:50]

    avg0 = ttrcs[motor,:,:].mean(axis=0)
    sem0 = sp.stats.sem(avg0)
    avg0 = avg0.mean(axis=0)
    avg1 = ttrcs[antimotor,:,:].mean(axis=0)
    sem1 = sp.stats.sem(avg1)
    avg1 = avg1.mean(axis=0)

    #avg0 = ttrcs[motor,:,:].mean(axis=1)
    #sem0 = sp.stats.sem(avg0)
    #avg0 = avg0.mean(axis=0)
    #avg1 = ttrcs[antimotor,:,:].mean(axis=1)
    #sem1 = sp.stats.sem(avg1)
    #avg1 = avg1.mean(axis=0)

    #avg0 = ttrcs[motor,:,:]
    #sem0 = sp.stats.sem(avg0, axis=(0,1))
    #avg0 = avg0.mean(axis=(0,1))
    #avg1 = ttrcs[antimotor,:,:]
    #sem1 = sp.stats.sem(avg1, axis=(0,1))
    #avg1 = avg1.mean(axis=(0,1))

    plt.figure()
    plt.title('Top and Bottom Neurons for Swim dF/F')
    plt.plot(tdrft[0,:], color='blue', label='drift')
    plt.plot(avg0, color='green', label='Top 50')
    plt.fill_between(np.arange(len(avg0)), avg0 - sem0, avg0 + sem0, color='green', alpha=0.3)
    plt.plot(avg1, color='magenta', label='Bottom 50')
    plt.fill_between(np.arange(len(avg1)), avg1 - sem1, avg1 + sem1, color='magenta', alpha=0.3)
    plt.legend()
    plt.savefig(join(odir, 'swim_top50_bottom50.png'), dpi=300)
    plt.close()

    #-------------------------------------------------------------------------------------------------------------#

    pcadir = os.path.join(odir, 'pca')
    os.makedirs(pcadir, exist_ok=True)

    _ = rois[3:11, 185:300, 175:295]
    inds = np.unique(_[_ > 0]) - 1
    print(len(inds), 'cells')

    temp = ttrcs[inds,:,:].mean(axis=1)

    pca = sk.decomposition.PCA(n_components=5)
    comps = pca.fit_transform(ttrcs[inds,:,:].mean(axis=1).T).T
    print(comps.shape)

    plt.figure()
    for i in range(len(comps)):
        plt.plot(comps[i,:], label=f'component {i}', alpha=0.3)
    plt.plot(tdrft[0,:], color='blue', label='drift')
    plt.legend()
    plt.savefig(os.path.join(pcadir, 'PCA_comps.png'), dpi=300)
    plt.close()

    #ica = sk.decomposition.FastICA(n_components=5)
    #comps = ica.fit_transform(ttrcs[inds,:,:].mean(axis=1).T).T
    #print(comps.shape)

    #plt.figure()
    #for i in range(len(comps)):
    #    plt.plot(comps[i,:], label=f'component {i}', alpha=0.3)
    #plt.plot(tdrft[0,:], color='blue', label='drift')
    #plt.legend()
    #plt.savefig(os.path.join(pcadir, 'ICA_comps.png'), dpi=300)
    #plt.close()


    plt.figure()
    plt.plot(comps[0,:]/5, label='component 0')
    plt.plot(avg1, color='green', label='slomo?')
    plt.fill_between(np.arange(len(avg1)), avg1 - sem1, avg1 + sem1, color='green', alpha=0.3)
    plt.plot(tdrft[0,:], label='drift')
    plt.legend()
    plt.savefig(os.path.join(pcadir, 'PCA_comps_compare.png'), dpi=300)
    plt.close()

    contributions = np.abs(pca.components_[0])
    indxd = np.argsort(contributions)

    plt.figure()
    plt.plot(comps[0,:]/5, label='component 0')
    #for i in range(5):
    #    plt.plot(ttrcs[indxd[-(i+1)], color='green', label='slomo?')
    plt.plot(tdrft[0,:], label='drift')
    plt.legend()
    plt.savefig(os.path.join(pcadir, 'PCA_comps_compare2.png'), dpi=300)
    plt.close()

    raise Exception
    _ = rois[3:11, 185:300, 175:295]
    #_ = rois[3:11, 185:300, 175:295]
    #_ = rois[4:11, 125:245, 250:360]
    #_ = rois[8:13, 160:265, 200:315]
    #group500 = np.intersect1d(_ - 1, indx[:500])
    #print(len(group500))
    #group250 = np.intersect1d(_ - 1, indx[:250])
    #print(len(group250))
    #group100 = np.intersect1d(_ - 1, indx[:100])
    #print(len(group100))
    #group50 = np.intersect1d(_ - 1, indx[:50])
    #print(len(group50))

    weights = np.zeros(Ln, np.float32)
    weights[group500] = 20
    drawn = lab.draw.draw_weights(rois, weights)
    tiff.imwrite(join(odir, 'swim_SLOMO.tif'), drawn, imagej=True, metadata={'axes':'ZYX'})

    #weights = np.zeros(Ln, np.float32)
    #weights[group] = 20
    #drawn = lab.draw.draw_weights(rois, weights)
    #tiff.imwrite(join(odir, 'swim_SLOMO.tif'), drawn, imagej=True, metadata={'axes':'ZYX'})
    
    groups = [group500, group250, group100, group50]
    plt.subplots(2,2)
    plt.title('SLOMO Neurons for Swim dF/F')
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.plot(tdrft[0,:], color='blue', label='drift')

        avg0 = ttrcs[groups[i],:,:].mean(axis=0)
        sem0 = sp.stats.sem(avg0)
        avg0 = avg0.mean(axis=0)

        plt.plot(avg0, color='green', label=f'SLOMO top {len(groups[i])}')
        plt.fill_between(np.arange(len(avg0)), avg0 - sem0, avg0 + sem0, color='green', alpha=0.3)

        plt.legend()
    plt.savefig(join(odir, 'swim_SLOMO.png'), dpi=300)
    plt.close()












    #trials = split_trials(expr)
    #dffs = (trcs - trcs.mean(axis=1, keepdims=True)) / trcs.std(axis=1, keepdims=True)

    #tbhvr = bhvr[trials]
    #tdrft = drft[trials]
    #tgain = gain[trials]
    #tswim = swim[trials]
    #tvlct = vlct[trials]
    #tfdbk = fdbk[trials]
    #ttrcs = dffs[:,trials]
    #assert tbhvr.shape == tdrft.shape == tgain.shape == tswim.shape == tvlct.shape == tfdbk.shape == ttrcs.shape[1:]
    #print(ttrcs.shape)

    #plt.figure()
    #plt.plot(tdrft[0,:], color='blue')
    #for i in range(10):
    #    #plt.plot(ttrcs[i,:,:].mean(axis=0))
    #    plt.plot(ttrcs[0,i,:])
    #plt.savefig(join(odir, 'test_this.png'), dpi=300)
    #plt.close()



    raise ValueError

    #print('[PROC] Computing timed variables')
    #swim_time = swim[:Lt]*range(Lt)
    #swim_timp = np.asarray(np.unique(swim_time)).astype(int)
    #drft_time = drft[:Lt]*range(Lt)
    #drft_timp = np.asarray(np.unique(drft_time)).astype(int)

    #nsnd_timp = np.asarray(range(Lt))[(~swim) & (drft==0)]
    #nsfd_timp = np.asarray(range(Lt))[(~swim) & (drft>0)]
    #nsbd_timp = np.asarray(range(Lt))[(~swim) & (drft<0)]

    #ysnd_timp = np.asarray(range(Lt))[(swim) & (drft==0)]
    #ysfd_timp = np.asarray(range(Lt))[(swim) & (drft>0)]
    #ysbd_timp = np.asarray(range(Lt))[(swim) & (drft<0)]

    #yswm_timp = np.asarray(range(Lt))[swim]
    #nswm_timp = np.asarray(range(Lt))[~swim]
    #print('[DONE] Timed variables computed')

    #-------------------------------------------------------------------------------------------------------------#


    motor_dffs = (trcs[:, swim].mean(axis=1) - trcs[:, ~swim].mean(axis=1))/trcs.mean(axis=1)
    optfd_dffs = (trcs[:, (~swim)&(drft>0)].mean(axis=1) - trcs[:, (~swim)&(drft==0)].mean(axis=1))/trcs.mean(axis=1)
    optbd_dffs = (trcs[:, (~swim)&(drft<0)].mean(axis=1) - trcs[:, (~swim)&(drft==0)].mean(axis=1))/trcs.mean(axis=1)
    fedbk_dffs = (trcs[:, (swim)&(drft>0)].mean(axis=1) - trcs[:, (swim)&(drft==0)].mean(axis=1))/trcs.mean(axis=1)

    drwn = lab.draw.draw_weights(rois, motor_dffs)
    tiff.imwrite(join(odir, 'dffs_motor.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    tiff.imwrite(join(odir, 'dffs_antimotor.tif'), -drwn, imagej=True, metadata={'axes':'ZYX'})
    drwn = lab.draw.draw_weights(rois, optfd_dffs)
    tiff.imwrite(join(odir, 'dffs_optfd.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    drwn = lab.draw.draw_weights(rois, optbd_dffs)
    tiff.imwrite(join(odir, 'dffs_optbd.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    drwn = lab.draw.draw_weights(rois, fedbk_dffs)
    tiff.imwrite(join(odir, 'dffs_fedbk.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})

    #-------------------------------------------------------------------------------------------------------------#

    perc = np.percentile(motor_dffs, 99.75)
    ngrp_motor = np.where(motor_dffs > perc)
    perc = np.percentile(motor_dffs, 0.25)
    ngrp_antimotor = np.where(motor_dffs < perc)

    perc = np.percentile(optfd_dffs, 99.75)
    ngrp_frwdoptic = np.where(optfd_dffs > perc)
    perc = np.percentile(optbd_dffs, 99.75)
    ngrp_bkwdoptic = np.where(optbd_dffs > perc)

    perc = np.percentile(fedbk_dffs, 99.75)
    ngrp_feedback = np.where(fedbk_dffs > perc)

    def temp_draw(wght, tag):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1,1]})

        bmgz = ax1.imshow(bmap.max(axis=0), cmap='gray', interpolation='nearest', vmax=np.percentile(bmap.max(axis=0).squeeze(), 99.9))
        wmgz = ax1.imshow(wght.max(axis=0), cmap='hot', interpolation='nearest', vmax=np.percentile(wght.max(axis=0).squeeze(), 99.9), alpha=0.5)

        bmgx = ax2.imshow(np.repeat(bmap.max(axis=1), 10, axis=0), cmap='gray', interpolation='nearest', vmax=np.percentile(bmap.max(axis=1).squeeze(), 99.9))
        wmgx = ax2.imshow(np.repeat(wght.max(axis=1), 10, axis=0), cmap='hot', interpolation='nearest', vmax=np.percentile(wght.max(axis=1).squeeze(), 99.9), alpha=0.5)
        plt.savefig(join(odir, f'testing_{tag}.png'), dpi=300)

    wght_motor = np.zeros(Ln, np.float32)
    wght_motor[ngrp_motor] = motor_dffs[ngrp_motor]
    drwn = lab.draw.draw_weights(rois, wght_motor)
    tiff.imwrite(join(odir, 'dffs_ngrp_motor.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    temp_draw(drwn, 'motor')

    wght_antimotor = np.zeros(Ln, np.float32)
    wght_antimotor[ngrp_antimotor] = np.abs(motor_dffs[ngrp_antimotor])
    drwn = lab.draw.draw_weights(rois, wght_antimotor)
    tiff.imwrite(join(odir, 'dffs_ngrp_antimotor.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    temp_draw(drwn, 'antimotor')

    wght_frwdoptic = np.zeros(Ln, np.float32)
    wght_frwdoptic[ngrp_frwdoptic] = optfd_dffs[ngrp_frwdoptic]
    drwn = lab.draw.draw_weights(rois, wght_frwdoptic)
    tiff.imwrite(join(odir, 'dffs_ngrp_frwdoptic.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    temp_draw(drwn, 'frwdoptic')

    wght_bkwdoptic = np.zeros(Ln, np.float32)
    wght_bkwdoptic[ngrp_bkwdoptic] = optbd_dffs[ngrp_bkwdoptic]
    drwn = lab.draw.draw_weights(rois, wght_bkwdoptic)
    tiff.imwrite(join(odir, 'dffs_ngrp_bkwdoptic.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    temp_draw(drwn, 'bkwdoptic')

    wght_feedback = np.zeros(Ln, np.float32)
    wght_feedback[ngrp_feedback] = fedbk_dffs[ngrp_feedback]
    drwn = lab.draw.draw_weights(rois, wght_feedback)
    tiff.imwrite(join(odir, 'dffs_ngrp_feedback.tif'), drwn, imagej=True, metadata={'axes':'ZYX'})
    temp_draw(drwn, 'feedback')

    #-------------------------------------------------------------------------------------------------------------#
    
    raise ValueError

    #if os.path.exists(os.path.join(odir, 'grph_MIcc.npz')):
    #    MIcc = sp.sparse.load_npz(os.path.join(odir, 'grph_MIcc.npz'))
    #else:
    #    MIcc = grph_MIcc(Ln, Lt)
    #    sp.sparse.save_npz(os.path.join(odir, 'grph_MIcc.npz'), MIcc)
    #print('MIcc graphed')

    if os.path.exists(os.path.join(odir, 'MI1v_bhvr.npy')):
        MIcc = sp.sparse.load_npz(os.path.join(odir, 'MI1v_bhvr.npz'))
    else:
        MIcc = grph_MIcc(Ln, Lt)
        sp.sparse.save_npz(os.path.join(odir, '.npz'), MIcc)
    print('MI1v computed')

    raise ValueError

    #-----------------------------------------------------------------------------------------#
    

    #-----------------------------------------------------------------------------------------#
    
    print('check1:', len(grph.row), 'check2', len(grph.col))
    print('visualizing weighted img')
    imgs = np.zeros((Ln, Ly, Lx), np.float32)
    for i in tqdm(range(Ln)):
        for j in range(Ln):
            if i == j:
                continue
            #wght = np.maximum(grph.data[(grph.row == i) & (grph.col == j)], grph.data[(grph.row == j) & (grph.col == i)])
            #wght = grph.data[(grph.row == i) & (grph.col == j)] + grph.data[(grph.row == j) & (grph.col == i)]
            mask = (grph.row == i) & (grph.col == j)
            mask2 = (grph.col == i) & (grph.row == j)
            mask3 = (mask|mask2)
            #mask *= (grph.col == j)
            #mask = ((grph.row == j) * (grph.col == i)) or ((grph.col == j) * (grph.row == i))

            wght = grph.data[mask3]
            imgs[i,:,:][rois == j+1] = wght
    cell = np.zeros((Ln, Ly, Lx), np.float32)
    for i in range(Ln):
        cell[i,:,:][rois == i+1] = 1.0

    np.save(os.path.join(odir, 'grph_wght_intr.npy'), imgs)
    
    cmap = LinearSegmentedColormap.from_list('black_to_green', ['black', 'green'])

    fig, ax = plt.subplots(frameon=True, dpi=300)
    plt.xlim(750,1500)
    plt.ylim(850,200)
    ax.axis('off')
    ax.set_axis_off()
    plt.margins(0)
    draw_bmap = ax.imshow(bmap, cmap='gray')
    draw_wght = ax.imshow(imgs[0], cmap='inferno', vmin=0, vmax=np.max(grph.data), alpha=0.2)
    draw_cell = ax.imshow(cell[0], cmap=cmap, vmin=0, vmax=1, alpha=0.3)
    #plt.savefig(os.path.join(pdir, 'st2p_ptch.pdf'), bbox_inches='tight', pad_inches=0)

    def update(nfrm):
        draw_wght.set_data(imgs[nfrm])
        draw_cell.set_data(cell[nfrm])
        return draw_wght, draw_cell

    movi = anim.FuncAnimation(
        fig,
        update,
        frames=Ln,
        interval=100,
        blit=True,
        repeat_delay=0,
        repeat=True
    )

    #movi.save(os.path.join(odir, 'grph_wght_movi.mp4'), writer='ffmpeg', fps=5)
    movi.save(os.path.join(odir, 'grph_wght_movi.gif'), writer='pillow', fps=10, dpi=300)
    plt.close(fig)





