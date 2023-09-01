import numpy as np
from brian2.only import *
import deepdish as dd

from digest import get_voltages
from readout import load_results

out_fname = 'contrib_TA.h5'


def get_contributions(cfg, isi, templ):
    '''
    Calculate the contribution of TA to greater activity in dev than msc, as a fraction of total contributions
    including TA and Vm. Contributions from STD are ignored.
    * Excludes bins (neurons x time) with less or equal activity in dev than msc
    * Considers negative contributions as zero (e.g., lower Vm does not contribute to more firing, thus contribution_Vm = 0)
    * Controls for magnitude of activity difference (delta pspike) by weighting
    * Averages across neurons
    Structure: {STD: ndarray((datasets, time))}
    '''
    print('get_contributions')
    TA = 1
    conds = 'dev', 'msc'
    measures = 'Vm', 'Threshold'
    contributions = {STD: {measure: [] for measure in measures} for STD in cfg.STDs}
    tmax_v = {STD: 0 for STD in cfg.STDs}
    for STD in cfg.STDs:
        print(f'STD {STD} ...')
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
                        for measure in measures:
                            hist[cond][measure] = V[measure]
                            tmax_v[STD] = max(tmax_v[STD], V[measure].shape[1])
                        hist[cond]['pspike'] = res['spikes'][ipair][stim][cond]['spike_hist']
                    
                    # Spike probability contrast
                    pdiff = hist['dev']['pspike'] - hist['msc']['pspike']
                    # Exclude entries with pspike(dev) <= pspike(msc)
                    mask = pdiff > 0
                    pdiff = np.where(mask, pdiff, np.nan)

                    # Voltage contrasts
                    dTA = hist['dev']['Threshold'] - hist['msc']['Threshold']
                    dV = hist['dev']['Vm'] - hist['msc']['Vm']

                    # Zero negative contribs and invert Threshold
                    dTA_driving = dTA * (dTA < 0) * -1
                    dV_driving = dV * (dV > 0)

                    # Take the ratio
                    dTA_contrib_full = dTA_driving / (dTA_driving + dV_driving)
                    dV_contrib_full = 1 - dTA_contrib_full

                    # Weight (and mask) and average across neurons
                    dTA_contrib_mean = np.nanmean(pdiff * dTA_contrib_full, axis=0)
                    dV_contrib_mean = np.nanmean(pdiff * dV_contrib_full, axis=0)

                    contributions[STD]['Threshold'].append(dTA_contrib_mean)
                    contributions[STD]['Vm'].append(dV_contrib_mean)
            print(net, end=' ', flush=True)
        for measure in measures:
            contributions[STD][measure] = np.asarray([
                np.concatenate([c, np.full_like(c, np.nan, shape=tmax_v[STD] - len(c))])
                for c in contributions[STD][measure]])
        print()
    
    return contributions


def process_to_disk(cfg, isi = None, templ = 0):
    if isi is None:
        isi = cfg.ISIs[0]
    contributions = get_contributions(cfg, isi, templ)
    dd.io.save(out_fname, contributions)


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'

    cfg = importlib.import_module(conf)
    process_to_disk(cfg)