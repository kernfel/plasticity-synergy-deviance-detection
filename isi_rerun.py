import sys
import importlib
import deepdish as dd
from brian2.only import *
import brian2genn

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import brian_cleanup

import isi as isipy

if __name__ == '__main__':
    cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))
    rng = np.random.default_rng()
        
    net = int(sys.argv[2])
    iISI = int(sys.argv[3])
    STD = int(sys.argv[4])
    TA = int(sys.argv[5])
    templ = 'R'

    res = dd.io.load(cfg.netfile.format(net=net))
    X, Y, W, D = res['X']*meter, res['Y']*meter, res['W'], res['D']
    Xstim, Ystim = spatial.create_stimulus_locations(cfg.params)

    assert 0 <= iISI < len(cfg.ISIs)
    isi = cfg.ISIs[iISI]

    assert STD in (0,1)
    assert TA in (0,1)

    del cfg.gpuid
    runit, working_dir = isipy.set_run_func(cfg)
    raw_fbase = cfg.raw_fbase if hasattr(cfg, 'raw_fbase') else None

    cfg.params = {**cfg.params, 'ISI': isi*ms,
                  'tau_rec': (0*ms, cfg.params['tau_rec'])[STD],
                  'th_ampl': (0*mV, cfg.params['th_ampl'])[TA],
                  'align_sequences': True}
    Net = model.create_network(X, Y, Xstim, Ystim, W, D, cfg.params, reset_dt=inputs.get_episode_duration(cfg.params))
    template = readout.setup_run(Net, cfg.params, rng, cfg.stimuli, cfg.pairings)

    rundata = runit(template, True, STD, TA, cfg.params, X, Y, Xstim, Ystim, W, D,
                    raw_fbase=None if raw_fbase is None else raw_fbase.format(**locals()))

    try:
        readout.save_results(cfg.fname.format(**locals()), rundata)
    except Exception as e:
        print("Error saving:", e)
    brian_cleanup(working_dir)
