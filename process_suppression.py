import numpy as np
from brian2.only import *
import deepdish as dd

from readout import load_results

out_fname = 'suppression.TA{TA}.h5'


def get_bspikes(res, episode):
    b = np.zeros(np.asarray(res['raw_dynamics']['v'].shape)[[0,2,3]], bool)
    for itrial, (i, spike_t) in enumerate(zip(*[res['raw_spikes'][episode][f'pulsed_{k}'] for k in 'it'])):
        t = (spike_t / res['params']['dt'] + .5).astype(int)
        b[i, itrial, t] = True
    return b


def get_suppression_data(res, Wb, ipair, istim, with_threshold=True, with_depression=True):
    Response = {AB: {} for AB in 'AB'}
    Threshold = {AB: {} for AB in 'AB'}
    Depression = {AB: {} for AB in 'AB'}
    
    target = res['pairs'][ipair][f'S{istim + 1}']
    nontarget = res['pairs'][ipair][f'S{(istim+1)%2 + 1}']

    # Response
    for cond in ('dev', 'msc', 'std'):
        episode = res['pairs'][ipair][cond][target]
        spikes = get_bspikes(res, episode)
        for isB, AB in enumerate('AB'):
            trials = res['sequences'][episode] == res['stimuli'][nontarget if isB else target]
            Response[AB][cond] = spikes[:, trials].sum(2).mean(1)
    
    # Suppression
    for cond in ('dev', 'msc'):
        episode = res['pairs'][ipair][cond][target]
        for isB, AB in enumerate('AB'):
            trials = res['sequences'][episode] == res['stimuli'][nontarget if isB else target]
            
            if with_threshold:
                Threshold[AB][cond] = res['raw_dynamics']['th_adapt'][:, episode, trials, 0].mean(1)*volt/mV
            
            if with_depression:
                xr = res['raw_dynamics']['neuron_xr'][:, episode, trials, 0].mean(1)
                Depression[AB][cond] = np.einsum('eo,e->o', Wb, np.mean([R for R in Response[AB].values()], 0) * (1-xr))
    
    return Response, Threshold, Depression


def get_suppression(cfg, isi, templ, TA):
    Rdata, TAdata, Ddata = [{STD: [] for STD in cfg.STDs} for _ in range(3)]
    print('get_suppression')
    for STD in cfg.STDs:
        print(f'STD {STD}, TA {TA} ...')
        for net in range(cfg.N_networks):
            res = load_results(
                cfg.fname.format(net=net, isi=isi, STD=STD, TA=TA, templ=templ),
                dynamics_supplements={'neuron_xr': 1},
                process_spikes=False, process_dynamics=False)
            netf = dd.io.load(cfg.netfile.format(net=net))
            Wb = netf['W'] > 0
            for ipair, pair in enumerate(cfg.pairings):
                for istim, stim in enumerate(pair):
                    data = get_suppression_data(res, Wb, ipair, istim, with_depression=STD, with_threshold=TA)
                    Rdata[STD].append(data[0])
                    TAdata[STD].append(data[1])
                    Ddata[STD].append(data[2])
            print(net, end=' ', flush=True)
        print()
    return Rdata, TAdata, Ddata


def process_to_disk(cfg, isi = None, templ = 0):
    if isi is None:
        isi = cfg.ISIs[0]

    R, T, D = get_suppression(cfg, isi, templ, 0)
    dd.io.save(out_fname.format(TA=0), {'R': R, 'D': D})

    R, T, D = get_suppression(cfg, isi, templ, 1)
    dd.io.save(out_fname.format(TA=1), {'R': R, 'T': T, 'D': D})


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'

    cfg = importlib.import_module(conf)
    process_to_disk(cfg)
