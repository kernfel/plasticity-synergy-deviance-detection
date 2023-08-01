import numpy as np
import deepdish as dd

from readout import load_results
from digest import conds

out_fname = 'nspikes.h5'


def get_nspikes(cfg, ISIs, templ):
    '''
    Retrieves the average number of spikes per trial, summed across the entire network.
    Structure: {STD: {TA: {cond: ndarray(nets*pairs*2)}}}
    '''
    nspikes = {isi: {STD: {TA: {cond: [] for cond in conds} for TA in cfg.TAs} for STD in cfg.STDs} for isi in ISIs}
    print('get_nspikes')
    for STD in cfg.STDs:
        for TA in cfg.TAs:
            for isi in ISIs:
                print(f'isi {isi}, STD {STD}, TA {TA} ...')
                for net in range(cfg.N_networks):
                    res = load_results(cfg.fname.format(net=net, isi=isi, STD=STD, TA=TA, templ=templ), compress=True, process_dynamics=False)
                    for ipair, pair in enumerate(cfg.pairings):
                        for istim, stim in enumerate(pair):
                            for cond in conds:
                                data = res['spikes'][ipair][stim][cond]
                                nspikes[isi][STD][TA][cond].append(data['nspikes'].sum(1).mean())
                    print(net, end=' ', flush=True)

                nspikes[isi][STD][TA] = {cond: np.asarray(x) for cond, x in nspikes[isi][STD][TA].items()}
                print()
    return nspikes


def process_to_disk(cfg, isi = None, templ = 0):
    if isi is None:
        isi = cfg.ISIs
    else:
        try:
            for _ in isi:
                break
        except ValueError:
            isi = [isi]
    nspikes = get_nspikes(cfg, isi, templ)
    dd.io.save(out_fname, nspikes)


if __name__ == '__main__':
    import sys
    import importlib

    if len(sys.argv) > 1:
        conf = '.'.join(sys.argv[1].split('.')[0].split('/'))
    else:
        conf = 'conf.isi5_500'

    cfg = importlib.import_module(conf)
    process_to_disk(cfg)
