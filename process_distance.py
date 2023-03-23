import numpy as np
from brian2.only import *
import deepdish as dd

from readout import load_results
from digest import conds
import spatial

out_fname = 'distance.h5'


def get_bspikes(res, episode):
    b = np.zeros(np.asarray(res['raw_dynamics']['v'].shape)[[0,2,3]], bool)
    for itrial, (i, spike_t) in enumerate(zip(*[res['raw_spikes'][episode][f'pulsed_{k}'] for k in 'it'])):
        t = (spike_t / res['params']['dt'] + .5).astype(int)
        b[i, itrial, t] = True
    return b


def get_neuron_distances(cfg, netf, stim):
    x,y = [xy[cfg.stimuli[stim]] for xy in spatial.create_stimulus_locations(cfg.params)]
    return np.sqrt((netf['X']*meter-x)**2 + (netf['Y']*meter-y)**2)


def get_trial_average(res, episode, stim_distances, stimidx=None):
    spike_distances, spike_counts, target = np.empty((3, len(res['sequences'][episode])))
    target = target.astype(bool)
    spikes = get_bspikes(res, episode)
    for trial, idx_stimulated in enumerate(res['sequences'][episode]):
        dist_from_stimulated = stim_distances[idx_stimulated]
        trial_spikes = spikes[:, trial]
        trial_distances = np.broadcast_to(dist_from_stimulated[:, None], trial_spikes.shape)[trial_spikes]
        spike_distances[trial] = np.median(trial_distances)
        target[trial] = idx_stimulated == stimidx
        spike_counts[trial] = trial_spikes.sum()
    return spike_distances, spike_counts, target


def get_distconf(cfg, K = 1.4, distbins = 200, rough_cutoff = 2*mm):
    maxdist = cfg.params['r_dish'] + cfg.params['stim_distribution_radius']
    
    histargs = {'bins': distbins, 'range': (0, maxdist/mm)}
    _, histedges = np.histogram([0], **histargs)

    histargs_rough = {'bins': [0, rough_cutoff/mm, maxdist/mm]}
    _, histedges_rough = np.histogram([0], **histargs_rough)
    
    return {'full': {'args': histargs, 'edges': histedges},
            'rough': {'args': histargs_rough, 'edges': histedges_rough},
            'K': K}


def get_trial_hist(distconf, res, episode, stim_distances):
    spikes = get_bspikes(res, episode)
    hist = np.empty((len(res['sequences'][episode]), distconf['full']['args']['bins']))
    hist_norm = {i: np.histogram(stim_distances[i]/mm, **distconf['full']['args'])[0] for i in np.unique(res['sequences'][episode])}
    rough = np.empty((len(res['sequences'][episode]), 2))
    rough_norm = {i: np.histogram(stim_distances[i]/mm, **distconf['rough']['args'])[0] for i in np.unique(res['sequences'][episode])}
    for trial, idx_stimulated in enumerate(res['sequences'][episode]):
        dist_from_stimulated = stim_distances[idx_stimulated]
        trial_spikes = spikes[:, trial]
        trial_distances = np.broadcast_to(dist_from_stimulated[:, None], trial_spikes.shape)[trial_spikes]
        trial_hist, edges = np.histogram(trial_distances/mm, **distconf['full']['args'])
        trial_rough, edges = np.histogram(trial_distances/mm, **distconf['rough']['args'])
        hist[trial] = trial_hist / hist_norm[idx_stimulated]
        rough[trial] = trial_rough / rough_norm[idx_stimulated]
    return hist, rough


def get_hist_base(distconf, stim_distances, stimid):
    hist, edges = np.histogram(stim_distances[stimid]/mm, **distconf['full']['args'])
    rough, edges = np.histogram(stim_distances[stimid]/mm, **distconf['rough']['args'])
    return hist, rough


def get_average_responses(cfg, res):
    R = {}
    allspikes = [get_bspikes(res, episode) for episode in range(len(res['sequences']))]
    for ipair, pair in enumerate(cfg.pairings):
        for istim, stim in enumerate(pair):
            stimidx = cfg.stimuli[stim]
            r = []
            for cond in conds:
                episode = res['pairs'][ipair][cond][stim]
                trials = res['sequences'][episode] == stimidx
                r.append(allspikes[episode][:, trials].sum(2).mean(1))
            R[stimidx] = np.mean(r, 0)
    return R


def get_distribution_indices(distconf, stim_distances, stim_indices):
    fine_binned_distances, rough_binned_distances = {}, {}
    fine_norm, rough_norm = {}, {}
    for idx in np.unique(stim_indices):
        dist = stim_distances[idx] / mm
        fine_binned_distances[idx] = np.digitize(dist, distconf['full']['edges']) - 1
        rough_binned_distances[idx] = np.digitize(dist, distconf['rough']['edges']) - 1
        fine_norm[idx], _ = np.histogram(dist, **distconf['full']['args'])
        rough_norm[idx], _ = np.histogram(dist, **distconf['rough']['args'])
    return fine_binned_distances, fine_norm, rough_binned_distances, rough_norm


def get_TA_distrib(distconf, res, episode, stim_distances):
    fine_TA = np.zeros((len(res['sequences'][episode]), distconf['full']['args']['bins']))
    rough_TA = np.zeros((len(res['sequences'][episode]), 2))
    fine_binned_distances, fine_norm, rough_binned_distances, rough_norm = get_distribution_indices(
        distconf, stim_distances, np.unique(res['sequences'][episode]))
    for trial, idx_stimulated in enumerate(res['sequences'][episode]):
        np.add.at(
            fine_TA[trial], fine_binned_distances[idx_stimulated],
            res['raw_dynamics']['th_adapt'][:, episode, trial, 0])
        fine_TA[trial] /= fine_norm[idx_stimulated]
        np.add.at(
            rough_TA[trial], rough_binned_distances[idx_stimulated],
            res['raw_dynamics']['th_adapt'][:, episode, trial, 0])
        rough_TA[trial] /= rough_norm[idx_stimulated]
    return fine_TA, rough_TA


def get_STD_distrib(distconf, res, episode, stim_distances, R, Wb):
    fine_STD = np.zeros((len(res['sequences'][episode]), distconf['full']['args']['bins']))
    rough_STD = np.zeros((len(res['sequences'][episode]), 2))
    fine_binned_distances, fine_norm, rough_binned_distances, rough_norm = get_distribution_indices(
        distconf, stim_distances, np.unique(res['sequences'][episode]))
    for trial, idx_stimulated in enumerate(res['sequences'][episode]):
        if idx_stimulated not in R:
            continue
        xr = res['raw_dynamics']['neuron_xr'][:, episode, trial, 0]
        D = np.einsum('eo,e->o', Wb, R[idx_stimulated] * (1-xr)) * distconf['K']
        np.add.at(fine_STD[trial], fine_binned_distances[idx_stimulated], D)
        fine_STD[trial] /= fine_norm[idx_stimulated]
        np.add.at(rough_STD[trial], rough_binned_distances[idx_stimulated], D)
        rough_STD[trial] /= rough_norm[idx_stimulated]
    return fine_STD, rough_STD


def get_distance(cfg, isi, templ, **kwargs):
    TA, STD = 1, 1
    distconf = get_distconf(cfg, **kwargs)

    disthist = {cond: np.full((cfg.N_networks*len(cfg.pairings)*2, cfg.params['sequence_length']*cfg.params['sequence_count'], distconf['full']['args']['bins']), np.nan)
                for cond in conds}
    distrough = {cond: np.full((cfg.N_networks*len(cfg.pairings)*2, cfg.params['sequence_length']*cfg.params['sequence_count'], 2), np.nan)
                for cond in conds}
    histbase = np.zeros((cfg.N_networks*len(cfg.pairings)*2, distconf['full']['args']['bins']))
    roughbase = np.zeros((cfg.N_networks*len(cfg.pairings)*2, 2))

    fine_TA, fine_STD = [
        {cond: np.full((cfg.N_networks*len(cfg.pairings)*2, cfg.params['sequence_length']*cfg.params['sequence_count'], distconf['full']['args']['bins']), np.nan)
        for cond in conds} for _ in 'ab']
    rough_TA, rough_STD = [
        {cond: np.full((cfg.N_networks*len(cfg.pairings)*2, cfg.params['sequence_length']*cfg.params['sequence_count'], 2), np.nan)
        for cond in conds} for _ in 'ab']

    print('get_distance ...')
    for net in range(cfg.N_networks):
        res = load_results(
            cfg.fname.format(net=net, isi=isi, STD=STD, TA=TA, templ=templ),
            dynamics_supplements={'neuron_xr': 1},
            process_spikes=False, process_dynamics=False)
        netf = dd.io.load(cfg.netfile.format(net=net))
        stim_distances = {iX: get_neuron_distances(cfg, netf, X) for X, iX in cfg.stimuli.items()}
        Wb = netf['W'] > 0
        R = get_average_responses(cfg, res)

        # msc
        msc_dist, msc_count, _ = get_trial_average(res, 0, stim_distances)
        msc_hist, msc_rough = get_trial_hist(distconf, res, 0, stim_distances)
        msc_fine_TA, msc_rough_TA = get_TA_distrib(distconf, res, 0, stim_distances)
        msc_fine_STD, msc_rough_STD = get_STD_distrib(distconf, res, 0, stim_distances, R, Wb)

        for ipair, pair in enumerate(cfg.pairings):
            for istim, stim in enumerate(pair):
                episode = res['pairs'][ipair]['dev'][stim]
                dist, count, target = get_trial_average(res, episode, stim_distances, cfg.stimuli[stim])
                hist, rough = get_trial_hist(distconf, res, episode, stim_distances)
                ep_fine_TA, ep_rough_TA = get_TA_distrib(distconf, res, episode, stim_distances)
                ep_fine_STD, ep_rough_STD = get_STD_distrib(distconf, res, episode, stim_distances, R, Wb)
                i = 4*net + 2*ipair + istim

                disthist['dev'][i, target] = hist[target]
                distrough['dev'][i, target] = rough[target]
                fine_TA['dev'][i, target] = ep_fine_TA[target]
                rough_TA['dev'][i, target] = ep_rough_TA[target]
                fine_STD['dev'][i, target] = ep_fine_STD[target]
                rough_STD['dev'][i, target] = ep_rough_STD[target]

                disthist['std'][i, ~target] = hist[~target]
                distrough['std'][i, ~target] = rough[~target]
                fine_TA['std'][i, ~target] = ep_fine_TA[~target]
                rough_TA['std'][i, ~target] = ep_rough_TA[~target]
                fine_STD['std'][i, ~target] = ep_fine_STD[~target]
                rough_STD['std'][i, ~target] = ep_rough_STD[~target]

                msc_target = res['sequences'][0] == cfg.stimuli[stim]
                disthist['msc'][i, msc_target] = msc_hist[msc_target]
                distrough['msc'][i, msc_target] = msc_rough[msc_target]
                fine_TA['msc'][i, msc_target] = msc_fine_TA[msc_target]
                rough_TA['msc'][i, msc_target] = msc_rough_TA[msc_target]
                fine_STD['msc'][i, msc_target] = msc_fine_STD[msc_target]
                rough_STD['msc'][i, msc_target] = msc_rough_STD[msc_target]

                histbase[i], roughbase[i] = get_hist_base(distconf, stim_distances, cfg.stimuli[stim])
        print(net, end=' ', flush=True)
    print()
    
    return {
        'disthist_initial': np.nanmean(np.concatenate([d[:, 0] for d in disthist.values()], 0), 0),
        'distrough_initial': np.nanmean(np.concatenate([d[:, 0] for d in distrough.values()], 0), 0),
        'disthist_steady': {cond: np.nanmean(disthist[cond][:, 1:], (0,1)) for cond in conds},
        'fine_TA_steady': {cond: np.nanmean(fine_TA[cond][:, 1:], (0,1)) for cond in conds},
        'fine_STD_steady': {cond: np.nanmean(fine_STD[cond][:, 1:], (0,1)) for cond in conds},
        'distrough_steady_nets': {cond: np.nanmean(distrough[cond][:, 1:], 1) for cond in conds},
        'rough_TA_steady_nets': {cond: np.nanmean(rough_TA[cond][:, 1:], 1) for cond in conds},
        'rough_STD_steady_nets': {cond: np.nanmean(rough_STD[cond][:, 1:], 1) for cond in conds},
        'histbase': histbase,
        'roughbase': roughbase,
        'distconf': distconf
    }




def process_to_disk(cfg, isi = None, templ = 0, **kwargs):
    if isi is None:
        isi = cfg.ISIs[0]
    distance = get_distance(cfg, isi, templ, **kwargs)
    dd.io.save(out_fname, distance)


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'

    cfg = importlib.import_module(conf)
    process_to_disk(cfg)