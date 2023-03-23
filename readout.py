import os
from distutils.log import warn
from importlib import import_module
from itertools import count
from brian2.only import *
from numpy.lib.format import open_memmap
import deepdish as dd

import numpy_ as np
import inputs

from util import ensure_unit
from spike_utils import iterspikes


def get_raw_spikes(Net, params, episodes):
    offset = 0
    npulses = params['sequence_length']*params['sequence_count']
    episodes = sorted(episodes)
    T, I = [{episode: [] for episode in episodes} for _ in range(2)]
    raw = {episode: {} for episode in episodes}
    for k in ['Exc'+Net.suffix, 'Inh'+Net.suffix]:
        all_t, all_i = Net[f'SpikeMon_{k}'].t, Net[f'SpikeMon_{k}'].i + offset
        offset += Net[k].N
        for episode, (episodic_i, episodic_t) in enumerate(iterspikes(
                all_i, all_t, max(episodes)+1, inputs.get_episode_duration(params), dt=params['dt'])):
            if episode in episodes:
                T[episode].append(episodic_t)
                I[episode].append(episodic_i)
    
    for episode in episodes:
        episodic_t = np.concatenate(T[episode]) - params['settling_period']
        idx = np.argsort(episodic_t)
        episodic_t = episodic_t[idx]
        episodic_i = np.concatenate(I[episode])[idx]
        raw[episode]['pulsed_i'], raw[episode]['pulsed_t'] = zip(*list(iterspikes(
            episodic_i, episodic_t, npulses, params['ISI'], dt=params['dt'])))
        raw[episode]['pulsed_nspikes'] = np.zeros((len(raw[episode]['pulsed_i']), params['N']), int)
        for j, i in enumerate(raw[episode]['pulsed_i']):
            np.add.at(raw[episode]['pulsed_nspikes'][j], i, 1)
    return raw


def raw_dynamics_filename(fbase, varname):
    return f'{fbase}_{varname}.npy'


def get_raw_dynamics(Net, params, episodes, tmax, raw_fbase=None):
    raw = {}
    npulses = params['sequence_length']*params['sequence_count']
    if 'StateMon_Exc'+Net.suffix in Net:
        t0_episode = (np.arange(max(episodes)+1) * inputs.get_episode_duration(params))[episodes]
        t0_pulse = np.arange(npulses)*params['ISI'] + params['settling_period']
        t_in_pulse = np.arange(stop=tmax, step=params['dt'])
        T = ((t0_episode[:, None, None] + t0_pulse[None, :, None] + t_in_pulse[None, None, :]) / params['dt'] + .5).astype(int)
        ones_inhibitory = np.ones((params['N_inh'],) + T.shape)
        storage_shape = (params['N'],) + T.shape

        for varname, init in Net['Exc'+Net.suffix].dynamic_variables.items():
            if not hasattr(Net['StateMon_Exc'+Net.suffix], varname):
                continue
            var_exc = getattr(Net['StateMon_Exc'+Net.suffix], varname)[:, T]
            try:
                var_inh = getattr(Net['StateMon_Inh'+Net.suffix], varname)[:, T]
            except AttributeError:
                if type(init) == str:
                    var_inh = ones_inhibitory * eval(init, globals(), params)
                else:
                    var_inh = ones_inhibitory * init
            
            if raw_fbase is not None:
                storage = open_memmap(raw_dynamics_filename(raw_fbase, varname), mode='w+', dtype=var_exc.dtype, shape=storage_shape)
            else:
                storage = np.empty(dtype=var_exc.dtype, shape=storage_shape)

            storage[:params['N_exc']] = var_exc
            storage[params['N_exc']:] = var_inh
            raw[varname] = storage
    return raw


def get_infused_histogram(params, spike_results, infusion, norm=False, default=0, **kwargs):
    '''
    Computes the pulse-triggered spike histogram of each neuron as a (N, t) ndarray,
    but instead of adding 1 for each spike, it adds `infusion(spike_results, pulse, indices, ticks, **kwargs)`,
    where `indices` and `ticks` are the firing neurons and corresponding time points
    in the given `pulse`.
    Note that this operates on **typed results**, and pulse indices thus enumerate the pulses
    of the given type (e.g. only A:deviant pulses)
    If `norm=False` (default), results are normalised by number of pulses.
    Otherwise, if `norm=True`, results are normalised by the number of spikes in each bin.
    '''
    hist = np.zeros((params['N'], int(params['ISI']/params['dt'] +.5)))
    hist_1 = np.zeros_like(hist)
    for pulse, (i, t) in enumerate(zip(spike_results['pulsed_i'], spike_results['pulsed_t'])):
        t = (t/params['dt'] +.5).astype(int)
        hist[i, t] += infusion(spike_results, pulse, i, t, **kwargs)
        hist_1[i, t] += 1
    if norm:
        np.divide(hist, hist_1, where=hist_1>0, out=hist)
    else:
        hist /= len(spike_results['pulsed_i'])
    hist[hist_1==0] = default
    return hist


def quantify_presynaptic(W, params, hist, xr):
    '''
    Finds:
    - The cumulative presynaptic input over STATIC weights (static_exc, static_inh: two (N,) vectors).
        In other words, the summed pulse-triggered spike histogram, weighted by static synaptic weight.
    - The factor by which static_exc is reduced due to short-term depression (depression_factor: one (N,) vector)
    Note: "input" here refers to the increments applied to the target's synaptic conductance.
    '''
    exc = np.zeros(params['N'], bool)
    exc[:params['N_exc']] = True
    W0 = W.copy()
    W0[np.isnan(W)] = 0
    # Postsyn input =   total presyn output       * weights,    summed over presyn
    static_exc = np.sum(hist[exc].sum(1)[:, None] * W0[exc, :], axis=0)
    static_inh = np.sum(hist[~exc].sum(1)[:, None] * W0[~exc, :], axis=0)
    dynamic = np.sum(xr[exc, None] * W0[exc, :], axis=0)
    return static_exc, static_inh, dynamic/static_exc


def separate_raw_spikes(rundata, params, compress=False, tmax=None):
    spike_output = []
    for pair in rundata['pairs']:
        out = {}
        spike_output.append(out)
        for S in (pair['S1'], pair['S2']):
            out[S] = {}
            for cond in ('std', 'dev', 'msc'):
                episode = pair[cond][S]
                raw = rundata['raw_spikes'][episode]
                pulse_mask = rundata['sequences'][episode] == rundata['stimuli'][S]
                results = out[S][cond] = {}
                
                results['nspikes'] = raw['pulsed_nspikes'][pulse_mask]
                results['pulsed_i'] = [i for i, j in zip(raw['pulsed_i'], pulse_mask) if j]
                results['pulsed_t'] = [i for i, j in zip(raw['pulsed_t'], pulse_mask) if j]
                results['spike_hist'] = get_infused_histogram(params, results, lambda *args: 1)
    
    if tmax is not None:
        itmax = int(tmax/params['dt'] + 0.5)
        compress = True
    elif compress:
        itmax = get_tmax(params, rundata['raw_spikes'])
    
    if compress:
        for out in spike_output:
            for S in out.keys():
                for key in out[S].keys():
                    out[S][key]['spike_hist'] = out[S][key]['spike_hist'][:, :itmax]
    
    return spike_output


def get_tmax(params, raw_spikes):
    itmax = 0
    for raw_episode in raw_spikes.values():
        for pulsed_t in raw_episode['pulsed_t']:
            itmax = max(itmax, int(pulsed_t[-1]/params['dt'] + .5))
    return itmax+1


def get_spike_results(Net, params, rundata, compress=False, tmax=None, processing=False):
    spike_output = []
    itmax = 0

    episodes = set()
    for pair in rundata['pairs']:
        for S in (pair['S1'], pair['S2']):
            for key in ('std', 'dev', 'msc'):
                episodes.add(pair[key][S])

    rundata['raw_spikes'] = get_raw_spikes(Net, params, list(episodes))
    
    if processing:
        rundata['spikes'] = separate_raw_spikes(rundata, params, compress, tmax)
    else:
        rundata['spikes'] = {}
    return rundata['spikes']


def get_episodes(rundata):
    episodes = set()
    for pair in rundata['pairs']:
        for S in (pair['S1'], pair['S2']):
            for cond in ('std', 'dev', 'msc'):
                episodes.add(pair[cond][S])
    return sorted(list(episodes))


def separate_raw_dynamics(rundata):
    episode_mapping = dict(zip(get_episodes(rundata), count()))
    dynamics = []
    for pair in rundata['pairs']:
        out = {}
        dynamics.append(out)
        for S in (pair['S1'], pair['S2']):
            out[S] = {}
            for cond in ('std', 'dev', 'msc'):
                episode = pair[cond][S]
                pulse_mask = rundata['sequences'][episode] == rundata['stimuli'][S]
                out[S][cond] = {}
                for varname in rundata['dynamic_variables']:
                    out[S][cond][varname] = rundata['raw_dynamics'][varname][:, episode_mapping[episode], pulse_mask]
    return dynamics


def get_dynamics_results(Net, params, rundata, compress=False, tmax=None, processing=False):
    if 'StateMon_Exc'+Net.suffix not in Net:
        return None

    if tmax is not None:
        compress = True
    elif compress:
        tmax = get_tmax(params, rundata['raw_spikes']) * params['dt']
    else:
        tmax = params['ISI']

    episodes = get_episodes(rundata)
    raw = get_raw_dynamics(Net, params, episodes, tmax, raw_fbase=rundata.get('raw_fbase', None))
    rundata['raw_dynamics'] = raw
    rundata['dynamic_variables'] = list(raw.keys())
    if processing:
        rundata['dynamics'] = separate_raw_dynamics(rundata)
    else:
        rundata['dynamics'] = {}
    return rundata['dynamics']


def get_results(Net, params, rundata, compress=False, tmax=None, processing=False):
    spike_output = get_spike_results(Net, params, rundata, compress=compress, tmax=tmax, processing=processing)
    dynamics_output = get_dynamics_results(Net, params, rundata, compress=compress, tmax=tmax, processing=processing)
    return spike_output, dynamics_output


def setup_run(Net, params, rng, stimuli, pairings=None):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d}).')
    stim_names = list(stimuli.keys())
    if pairings is None:
        pairings = [(stim_names[2*i], stim_names[2*i+1]) for i in range(len(stim_names)//2)]
    assert len(pairings)
    
    MSC, T = inputs.create_MSC(Net, params, rng)
    episode = 1
    sequences = [MSC]
    pairs = []
    for S1, S2 in pairings:
        if params.get('align_sequences', False):
            for S in (S1, S2):
                oddball = MSC.copy()
                nontargets = (oddball != stimuli[S1]) & (oddball != stimuli[S2])
                oddball[nontargets] = stimuli[S]
                T = inputs.set_input_sequence(Net, oddball, params, offset=T)
                sequences.append(oddball)
        else:
            oddball1, T = inputs.create_oddball(
                Net, params, stimuli[S1], stimuli[S2], rng, offset=T)
            oddball2, T = inputs.create_oddball(
                Net, params, stimuli[S2], stimuli[S1], rng, offset=T)
            sequences.extend([oddball1, oddball2])
        pairs.append({
            'S1': S1, 'S2': S2,
            'msc': {S1: 0, S2: 0},
            'std': {S1: episode, S2: episode+1},
            'dev': {S1: episode+1, S2: episode}})
        episode += 2
    return {'sequences': sequences, 'pairs': pairs, 'runtime': T, 'stimuli': stimuli, 'params': params}


def repeat_run(Net, params, template):
    d = inputs.get_episode_duration(params)
    if Net.reset_dt != d:
        warn(f'Net reset_dt ({Net.reset_dt}) does not match episode duration ({d}).')
    T = 0*second
    for seq in template['sequences']:
        T = inputs.set_input_sequence(Net, seq, params, T)
    return {**{k: template[k] for k in ('sequences', 'pairs', 'stimuli')}, 'runtime': T, 'params': params}


def save_results(fname, rundata):
    elide = ['dynamics', 'spikes']
    if rundata.get('raw_fbase', None) is not None:
        elide += ['raw_dynamics']
    os.makedirs(os.path.dirname(os.path.abspath(fname)), exist_ok=True)
    dd.io.save(fname, {k:v for k,v in rundata.items() if k not in elide})


def load_results(fname, dynamics_supplements={}, raw_fbase=None, compress=False, tmax=None,
                 process_spikes=True, process_dynamics=True, template_only=False):
    if template_only:
        template_keys = 'sequences', 'pairs', 'runtime', 'stimuli', 'params'
        return dict(zip(template_keys, dd.io.load(fname, [f'/{key}' for key in template_keys])))
    rundata = dd.io.load(fname)
    rundata['raw_fbase'] = raw_fbase or rundata.get('raw_fbase', None)
    if rundata['raw_fbase'] is not None and 'raw_dynamics' not in rundata:
        rundata['raw_dynamics'] = {}
        for varname in rundata['dynamic_variables']:
            rundata['raw_dynamics'][varname] = open_memmap(raw_dynamics_filename(rundata['raw_fbase'], varname), mode='r')
    for k, v in dynamics_supplements.items():
        if k in rundata['raw_dynamics']:
            continue
        elif type(v) == str and v in rundata['raw_dynamics']:
            rundata['raw_dynamics'][k] = rundata['raw_dynamics'][v]
        elif isinstance(v, np.ndarray):
            rundata['raw_dynamics'][k] = v
        else:
            rundata['raw_dynamics'][k] = np.ones_like(np.array(rundata['raw_dynamics'][rundata['dynamic_variables'][0]])) * v
        rundata['dynamic_variables'].append(k)
    default_params = import_module('conf.params').params
    param_units = {k: get_unit(v.dim) for k,v in default_params.items() if isinstance(v, Quantity)}
    rundata['params'] = ensure_unit(rundata['params'], param_units)
    if process_spikes:
        for s_episode in rundata['raw_spikes'].values():
            for s_trial in s_episode['pulsed_t']:
                s_trial[:] = ensure_unit(s_trial, second)
        rundata['spikes'] = separate_raw_spikes(rundata, rundata['params'], compress, tmax)
    if process_dynamics:
        rundata['dynamics'] = separate_raw_dynamics(rundata)
    return rundata
