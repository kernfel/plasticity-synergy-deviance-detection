import sys
import importlib
from collections import defaultdict
import warnings

from matplotlib.pyplot import hist
from numpy.lib.format import open_memmap

from brian2.only import *
import deepdish as dd

import numpy_ as np
from util import Tree, ensure_unit
from readout import load_results


conds = ('std', 'msc', 'dev')
voltage_measures = ('Activity', 'Depression', 'Threshold')


def get_voltages(params, dynamics, overflow=None):
    depression = ensure_unit(dynamics['u'] - dynamics['v'], volt)
    threshold = ensure_unit(dynamics['th_adapt'], volt)
    vm = ensure_unit(dynamics['v'], volt)
    return {
        'Depression': depression,
        'Threshold': threshold,
        'Vm': vm}


def get_voltage_histograms(params, rundata, overflow=None):
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', message='Mean of empty slice')
        
        hists = Tree()
        for ipair, pair in enumerate(rundata['pairs']):
            for stim in (pair['S1'], pair['S2']):
                for cond in conds:
                    dynamics = rundata['dynamics'][ipair][stim][cond]
                    voltages = get_voltages(params, dynamics, overflow)
                    for measure, val in voltages.items():
                        hists[measure][ipair][stim][cond] = val.mean(1)
        return hists.asdict()


def iter_runs(cfg, dynamics_only=False):
    N_templates = min(cfg.N_templates, cfg.N_templates_with_dynamics) if dynamics_only else cfg.N_templates
    for templ in range(N_templates):
        for net in range(cfg.N_networks):
            for STD in cfg.STDs:
                for TA in cfg.TAs:
                    for iISI, isi in enumerate(cfg.ISIs):
                        yield templ, net, STD, TA, iISI, isi


def get_digest_output(cfg, kind):
    if kind in ('nspikes-pulsemean', 'nspikes-neuronmean'):
        return dd.io.load(cfg.digestfile.format(kind=kind) + '.h5')
    elif kind == 'nspikes':
        keys = conds + ('nontarget_msc',)
    elif kind == 'histograms':
        keys = voltage_measures + ('pspike',)
    elif kind == 'masked_histograms':
        keys = voltage_measures + ('weight',)
    return {key: open_memmap(cfg.digestfile.format(kind=f'{kind}-{key}') + '.npy', 'c') for key in keys}


def digest(cfg, spikes=True, hist=True, masked=True):
    assert spikes or hist or masked
    spike_runs_shape = (cfg.N_templates, cfg.N_networks, len(cfg.STDs), len(cfg.TAs), len(cfg.ISIs), len(cfg.pairings), 2)
    dynamic_runs_shape = (min(cfg.N_templates, cfg.N_templates_with_dynamics),) + spike_runs_shape[1:] + (len(conds),)
    nspikes = {}
    histograms, masked_histograms = None, None
    log_nbins = np.log2(cfg.params['ISI']/cfg.params['dt'])
    nspikes_dtype = np.int8 if log_nbins < 7 else np.int16 if log_nbins < 15 else np.int32 if log_nbins < 31 else np.int64
    for templ, net, STD, TA, iISI, isi in iter_runs(cfg):
        try:
            res = load_results(cfg.fname.format(**locals()))
        except Exception as e:
            print(e)
        for ipair, pair in enumerate(res['pairs']):
            for istim, stim in enumerate((pair['S1'], pair['S2'])):
                for icond, cond in enumerate(conds):
                    idx = templ, net, STD, TA, iISI, ipair, istim, icond
                    thespikes = res['spikes'][ipair][stim][cond]
                    if spikes:
                        if cond not in nspikes:
                            nspikes[cond] = open_memmap(
                                cfg.digestfile.format(kind=f'nspikes-{cond}') + '.npy', dtype=nspikes_dtype, mode='w+',
                                shape=spike_runs_shape + thespikes['nspikes'].shape)
                        nspikes[cond][idx[:-1]] = thespikes['nspikes']

                    if hist and 'voltage_histograms' in res:
                        for measure, hists in res['voltage_histograms'].items():
                            thehist = hists[ipair][stim][cond]
                            if histograms is None:
                                histograms = {
                                    k: open_memmap(
                                        cfg.digestfile.format(kind=f'histograms-{k}') + '.npy', dtype=float, mode='w+',
                                        shape=dynamic_runs_shape + thehist.shape)
                                    for k in list(res['voltage_histograms'].keys()) + ['pspike']}
                            histograms[measure][idx] = thehist
                        histograms['pspike'][idx] = thespikes['spike_hist']

                    if masked and 'masked_voltage_histograms' in res:
                        for measure, hists in res['masked_voltage_histograms'].items():
                            thehist = hists[ipair][stim][cond]
                            if masked_histograms is None:
                                masked_histograms = {
                                    k: open_memmap(
                                        cfg.digestfile.format(kind=f'masked_histograms-{k}') + '.npy', dtype=float, mode='w+',
                                        shape=dynamic_runs_shape + thehist.shape)
                                    for k in res['masked_voltage_histograms'].keys()}
                            masked_histograms[measure][idx] = thehist
                
                if spikes:
                    episode = pair['msc'][stim]
                    pulse_mask = res['sequences'][episode] != res['stimuli'][stim]  # Non-target MSC pulses
                    nontarget_nspikes = res['raw_spikes'][episode]['pulsed_nspikes'][pulse_mask]

                    cond = 'nontarget_msc'
                    if cond not in nspikes:
                        nspikes[cond] = open_memmap(
                            cfg.digestfile.format(kind=f'nspikes-{cond}') + '.npy', dtype=nspikes_dtype, mode='w+',
                            shape=spike_runs_shape + nontarget_nspikes.shape)
                    nspikes[cond][idx[:-1]] = nontarget_nspikes

    if histograms is not None:
        scrub_stimulated_overactivation(cfg, histograms)
    if masked_histograms is not None:
        scrub_stimulated_overactivation(cfg, masked_histograms)
    if spikes:
        try:
            dd.io.save(cfg.digestfile.format(kind='nspikes-pulsemean') + '.h5', {cond: n.mean(-2) for cond, n in nspikes.items()})
            dd.io.save(cfg.digestfile.format(kind='nspikes-neuronmean') + '.h5', {cond: n.mean(-1) for cond, n in nspikes.items()})
        except Exception as e:
            print(e)


def scrub_stimulated_overactivation(cfg, histograms):
    for net in range(cfg.N_networks):
        try:
            stimulated = dd.io.load(cfg.netfile.format(net=net))['stimulated_neurons']
        except Exception as e:
            print(e)
        for ipair, pair in enumerate(cfg.pairings):
            for istim, stim in enumerate(pair):
                stimulated_neurons = stimulated[cfg.stimuli[stim]]
                unstimulated_neurons = np.flatnonzero(~np.isin(np.arange(cfg.params['N']), stimulated_neurons))
                idx = slice(None), net, slice(None), slice(None), slice(None), ipair, istim, slice(None)
                idx_to_filter = *idx, stimulated_neurons
                idx_reference = *idx, unstimulated_neurons
                reference = np.nanmax(histograms['Activity'][idx_reference])
                histograms['Activity'][idx_to_filter] = np.minimum(histograms['Activity'][idx_to_filter], reference)


if __name__ == '__main__':
    if '-p' in sys.argv:
        sys.argv.remove('-p')
        piecemeal = True
    elif '--piecemeal' in sys.argv:
        sys.argv.remove('--piecemeal')
        piecemeal = True
    else:
        piecemeal = False
    cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))
    if piecemeal:
        digest(cfg, spikes=True, hist=False, masked=False)
        digest(cfg, spikes=False, hist=True, masked=False)
        digest(cfg, spikes=False, hist=False, masked=True)
    else:
        digest(cfg)
