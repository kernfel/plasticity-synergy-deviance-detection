import numpy as np
from brian2.only import *
import deepdish as dd

from digest import conds, get_voltages
from readout import load_results

out_fname = 'contrasts.h5'


def get_contrasts(cfg, isi, templ):
    '''
    Calculate voltage measure contrasts (dev - msc), weighted by pspike contrast (dev - msc),
    and masked by pspike contrast (dev - msc > 0.05)
    Structure: {STD: {TA: {measure: ndarray((nets*pairs*2, time))}}}
    '''
    measures = ('Vm', 'Depression', 'Threshold')
    contrasts = {STD: {TA: {measure: [] for measure in measures} for TA in cfg.TAs} for STD in cfg.STDs}
    tmax_v = {STD: {TA: 0 for TA in cfg.TAs} for STD in cfg.STDs}
    print('get_contrasts')
    for STD in cfg.STDs:
        for TA in cfg.TAs:
            print(f'STD {STD}, TA {TA} ...')
            for net in range(cfg.N_networks):
                res = load_results(
                    cfg.fname.format(net=net, isi=isi, STD=STD, TA=TA, templ=templ),
                    compress=True, dynamics_supplements={'u': 'v', 'th_adapt': 0})
                for ipair, pair in enumerate(cfg.pairings):
                    for istim, stim in enumerate(pair):
                        hist = {cond: {} for cond in conds}
                        for cond in conds:
                            data = res['dynamics'][ipair][stim][cond]
                            # Average over trials, leaving measures shaped (neuron, time):
                            V = {measure: v.mean(1) for measure, v in get_voltages(cfg.params, data).items()}
                            for measure, v in V.items():
                                hist[cond][measure] = v
                                tmax_v[STD][TA] = max(tmax_v[STD][TA], v.shape[1])
                            hist[cond]['pspike'] = res['spikes'][ipair][stim][cond]['spike_hist']
                        
                        # Spike probability contrast
                        pdiff = hist['dev']['pspike'] - hist['msc']['pspike']
                        # Mask out entries where the pdiff contrast <= 0.05
                        pdiff = np.where(pdiff > .05, pdiff, np.nan)

                        # Weight & mask measures by pspike
                        for measure in measures:
                            # Calculate raw contrast (dev - msc)
                            c = (hist['dev'][measure] - hist['msc'][measure])/mV
                            # Weight with pdiff contrast, and sum valid entries over neurons
                            contrasts[STD][TA][measure].append(np.nansum(c*pdiff, 0))
                print(net, end=' ', flush=True)
            
            # Stack everything in {STD: {TA: {measure: ...}}} into a (nets*pairs*2, tmax_v) ndarray
            contrasts[STD][TA] = {
                measure: np.asarray([
                    np.concatenate([c, np.full_like(c, np.nan, shape=tmax_v[STD][TA]-len(c))])
                    for c in contrast
                ])
                for measure, contrast in contrasts[STD][TA].items()}
            print()
    
    return contrasts


def process_to_disk(cfg, isi = None, templ = 0):
    if isi is None:
        isi = cfg.ISIs[0]
    contrasts = get_contrasts(cfg, isi, templ)
    dd.io.save(out_fname, contrasts)


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'

    cfg = importlib.import_module(conf)
    process_to_disk(cfg)