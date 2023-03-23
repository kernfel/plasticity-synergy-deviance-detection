import numpy as np
from brian2.only import *
import deepdish as dd

from digest import conds
import isi_indexing as ii
from readout import load_results

out_fname = 'stats.h5'


def get_bspikes(res, episode):
    b = np.zeros(np.asarray(res['raw_dynamics']['v'].shape)[[0,2,3]], bool)
    for itrial, (i, spike_t) in enumerate(zip(*[res['raw_spikes'][episode][f'pulsed_{k}'] for k in 'it'])):
        t = (spike_t / res['params']['dt'] + .5).astype(int)
        b[i, itrial, t] = True
    return b


def get_stats(cfg, isi, templ, early=100):
    '''
    Computes a bunch of stats across networks and stimuli.
    Requires that process_suppression.process_to_disk has been run with the same args.
    Args:
        early: int, number of neurons to consider in the "early" subpopulation
    '''
    Astim = [stim for net in range(cfg.N_networks) for pair in cfg.pairings for stim in pair]
    Bstim = [stim for net in range(cfg.N_networks) for pair in cfg.pairings for stim in pair[::-1]]
    Aidx = np.asarray([cfg.stimuli[stim] for stim in Astim])
    Bidx = np.asarray([cfg.stimuli[stim] for stim in Bstim])

    TAdata = dd.io.load('suppression.TA1.h5', '/T')
    Ddata = {0: dd.io.load('suppression.TA0.h5', '/D')[1], 1: dd.io.load('suppression.TA1.h5', '/D')[1]}
    
    shape = cfg.N_networks*len(cfg.pairings)*2, cfg.params['N'], cfg.params['sequence_count']*cfg.params['sequence_length']
    dXnotA, dXA, stimchange, adaptation = [{STD: {TA: np.empty(shape[:2]) for TA in cfg.TAs} for STD in cfg.STDs} for _ in range(4)]
    dXnotA_early, dXnotA_late = [{STD: {TA: np.empty(shape[0]) for TA in cfg.TAs} for STD in cfg.STDs} for _ in range(2)]
    dTA_B = {STD: np.empty(shape[:2]) for STD in cfg.STDs}
    dTA_B_early, dTA_B_late = [{STD: np.empty(shape[0]) for STD in cfg.STDs} for _ in range(2)]
    dD_B = {TA: np.empty(shape[:2]) for TA in cfg.TAs}
    dD_B_early, dD_B_late = [{TA: np.empty(shape[0]) for TA in cfg.TAs} for _ in range(2)]
    print('get_stats')
    for STD in cfg.STDs:
        for TA in cfg.TAs:
            print(f'STD {STD}, TA {TA} ...')
            for net in range(cfg.N_networks):
                res = load_results(cfg.fname.format(net=net, isi=isi, STD=STD, TA=TA, templ=templ), compress=True, process_dynamics=False)
                bspikes_ep = {}
                for ipair, pair in enumerate(cfg.pairings):
                    for istim, stim in enumerate(pair):
                        k = 2*len(cfg.pairings)*net + 2*ipair + istim
                        bspikes = {}
                        A, B = Aidx[k], Bidx[k]
                        ep = {cond: res['pairs'][ipair][cond][stim] for cond in conds}
                        seq = {cond: res['sequences'][ep[cond]] for cond in conds}
                        for cond, episode in ep.items():
                            if episode not in bspikes_ep:
                                bspikes_ep[episode] = get_bspikes(res, episode)
                            bspikes[cond] = bspikes_ep[episode]

                        dXnotA[STD][TA][k] = (
                            bspikes['dev'].sum(2)[:, seq['dev']!=A].mean(1)
                            - bspikes['msc'].sum(2)[:, seq['msc']!=A].mean(1))
                        dXA[STD][TA][k] = (
                            bspikes['dev'].sum(2)[:, seq['dev']==A].mean(1)
                            - bspikes['msc'].sum(2)[:, seq['msc']==A].mean(1))

                        stimchange[STD][TA][k] = (
                            bspikes['msc'].sum(2)[:, seq['msc']==B].mean(1)
                            - bspikes['msc'].sum(2)[:, seq['msc']!=A].mean(1))
                        adaptation[STD][TA][k] = (
                            bspikes['dev'].sum(2)[:, seq['dev']==B].mean(1)
                            - bspikes['msc'].sum(2)[:, seq['msc']==B].mean(1))
                        
                        index_N, index_t = ii.get_onset_ordering(cfg, np.stack([b[:, seq[cond]==B, :].mean(1) for cond, b in bspikes.items()]))
                        early_B, late_B = index_N[:early, 0], index_N[early:, 0]
                        dXnotA_early[STD][TA][k] = np.median(
                            bspikes['dev'][early_B][:, seq['dev']!=A].sum(2).mean(1)
                            - bspikes['msc'][early_B][:, seq['msc']!=A].sum(2).mean(1))
                        dXnotA_late[STD][TA][k] = np.median(
                            bspikes['dev'][late_B][:, seq['dev']!=A].sum(2).mean(1)
                            - bspikes['msc'][late_B][:, seq['msc']!=A].sum(2).mean(1))
                        
                        if TA:
                            dVTA = TAdata[STD][k]['B']['dev'] - TAdata[STD][k]['B']['msc']
                            dTA_B[STD][k] = dVTA
                            dTA_B_early[STD][k] = np.median(dVTA[early_B])
                            dTA_B_late[STD][k] = np.median(dVTA[late_B])
                        
                        if STD:
                            dD = Ddata[TA][k]['B']['dev'] - Ddata[TA][k]['B']['msc']
                            dD_B[TA][k] = dD
                            dD_B_early[TA][k] = np.median(dD[early_B])
                            dD_B_late[TA][k] = np.median(dD[late_B])
                print(net, end=' ', flush=True)
            print()
    dTA_A = {STD: [] for STD in cfg.STDs}
    for STD in cfg.STDs:
        for k, ta in enumerate(TAdata[STD]):
            dTA_A[STD].append(ta['A']['dev'] - ta['A']['msc'])
        dTA_A[STD] = np.asarray(dTA_A[STD])
    return {
        'stimchange': stimchange,
        'adaptation': adaptation,
        'dXnotA': dXnotA,
        'dXA': dXA,
        'dTA_A': dTA_A,
        'dXnotA_early': dXnotA_early,
        'dXnotA_late': dXnotA_late,
        'dTA_B': dTA_B,
        'dTA_B_early': dTA_B_early,
        'dTA_B_late': dTA_B_late,
        'dD_B': dD_B,
        'dD_B_early': dD_B_early,
        'dD_B_late': dD_B_late}


def process_to_disk(cfg, isi = None, templ = 0):
    if isi is None:
        isi = cfg.ISIs[0]
    stats = get_stats(cfg, isi, templ)
    dd.io.save(out_fname, stats)


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'

    cfg = importlib.import_module(conf)
    process_to_disk(cfg)
