import time
import sys
import os
import functools
import importlib
import multiprocessing as mp
import deepdish as dd
from brian2.only import *
import brian2genn

# for the IDE:
import numpy_ as np
import spatial, model, inputs, readout

from util import brian_cleanup
from digest import get_voltage_histograms


def run_cpu(cfg, template, with_dynamics, STD, TA, mod_params, *net_args, raw_fbase=None, **device_args):
    device.reinit()
    device.activate(**device_args)

    if with_dynamics:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params),
            state_dt=cfg.params['dt'],
            state_vars=['v', 'vsyn'] + [k for k,v in (('th_adapt', TA), ('u', STD), ('neuron_xr', STD)) if v])
    else:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params))
    
    rundata = readout.repeat_run(Net, mod_params, template)
    rundata['raw_fbase'] = raw_fbase
    Net.run(rundata['runtime'])
    readout.get_results(Net, mod_params, rundata, compress=True, tmax=cfg.ISIs[0]*ms if len(cfg.ISIs)>1 else None)
    return rundata


def run_genn(cfg, template, with_dynamics, STD, TA, mod_params, *net_args, raw_fbase=None, **device_args):
    device.reinit()
    device.activate(**device_args)
    
    if with_dynamics:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params),
            state_dt=cfg.params['dt'], state_vars=['v', 'vsyn'] + [k for k,v in (('th_adapt', TA), ('neuron_xr', STD)) if v])
    else:
        Net = model.create_network(
            *net_args, params=mod_params,
            reset_dt=inputs.get_episode_duration(mod_params))
    
    rundata = readout.repeat_run(Net, mod_params, template)
    rundata['raw_fbase'] = raw_fbase
    Net.run(rundata['runtime'])
    readout.get_results(Net, mod_params, rundata, compress=True, tmax=cfg.ISIs[0]*ms if len(cfg.ISIs)>1 else None)

    if STD and with_dynamics:
        surrogate = {k: {'t': Net[f'SpikeMon_{k}'].t[:], 'i': Net[f'SpikeMon_{k}'].i[:]} for k in ('Exc', 'Inh')}

        device.reinit()
        device.activate(**device_args)
        
        mod_params_U = {**mod_params, 'tau_rec': 0*ms}
        Net = model.create_network(
            *net_args, params=mod_params_U,
            reset_dt=inputs.get_episode_duration(mod_params_U),
            state_dt=cfg.params['dt'], state_vars=['v'],
            surrogate=surrogate, suffix='_surrogate')
        
        rundata_U = readout.repeat_run(Net, mod_params_U, template)
        rundata_U['raw_fbase'] = None if raw_fbase is None else f'utmp.{raw_fbase}'
        Net.run(rundata_U['runtime'])
        readout.get_results(Net, mod_params_U, rundata_U, compress=True, tmax=cfg.ISIs[0]*ms)
        for V_pair, U_pair in zip(rundata['dynamics'], rundata_U['dynamics']):
            for S in V_pair.keys():
                for tp in V_pair[S].keys():
                    V_pair[S][tp]['u'] = U_pair[S][tp]['v']
        rundata['raw_dynamics']['u'] = rundata_U['raw_dynamics']['v']
        rundata['dynamic_variables'].append('u')
        if raw_fbase is not None:
            vfile = readout.raw_dynamics_filename(rundata_U['raw_fbase'], 'v')
            ufile = readout.raw_dynamics_filename(rundata['raw_fbase'], 'u')
            os.replace(vfile, ufile)
    return rundata


def generate_network(cfg, rng):
    params = cfg.params.copy()
    params['ISI'] = cfg.stim_probe_duration if 'stim_probe_duration' in dir(cfg) else params.get('ISI', 100*ms)
    params['sequence_length'], params['sequence_count'] = 1, 1
    min_frac = cfg.minimum_active_fraction if 'minimum_active_fraction' in dir(cfg) else 0.5
    Xstim, Ystim = spatial.create_stimulus_locations(params)
    while True:
        X, Y, W, D = spatial.create_weights(params, rng)
        Net = model.create_network(X, Y, Xstim, Ystim, W, D, params, reset_dt=params['ISI'] + params['settling_period'])
        offset = 0*second
        for i in range(params['N_stimuli']):
            offset = inputs.set_input_sequence(Net, [i], params, offset)
        Net.run(offset)
        spikes = readout.get_raw_spikes(Net, params, range(params['N_stimuli']))
        frac = [(ep['pulsed_nspikes']>0).mean() for ep in spikes.values()]
        sufficient_activity = all([f > min_frac for f in frac])
        print(f'{sufficient_activity}\t{frac}')
        if sufficient_activity:
            return X, Y, W, D, Xstim, Ystim


def generate_networks(cfg, rng, start_at):
    if start_at.get('templ', 0) > 0:
        return
    warned_about_premades = False
    for net in range(cfg.N_networks):
        if net < start_at.get('net', 0):
            continue
        elif start_at.get('newnet', True):
            if 'nets' in dir(cfg) and net in cfg.nets:
                res = dd.io.load(cfg.nets[net])
                X, Y, W, D = res['X']*meter, res['Y']*meter, res['W'], res['D']
                if not warned_about_premades:
                    print(f'Note: Config-supplied networks are not checked for response magnitude.')
                    warned_about_premades = True
            else:
                X, Y, W, D, Xstim, Ystim = generate_network(cfg, rng)
            
            stimulated_neurons = spatial.get_stimulated(X, Y, Xstim, Ystim, cfg.params)
            try:
                dd.io.save(cfg.netfile.format(net=net), dict(X=X, Y=Y, W=W, D=D, stimulated_neurons=stimulated_neurons))
            except Exception as e:
                print(e)


def set_run_func(cfg):
    if 'gpuid' in dir(cfg):
        working_dir = f'tmp/GPU{cfg.gpuid}'
        dev = 'genn'
        prefs.devices.genn.cuda_backend.device_select='MANUAL'
        prefs.devices.genn.cuda_backend.manual_device=cfg.gpuid
        runit = run_genn
    else:
        working_dir = 'tmp/CPP'
        dev = 'cpp_standalone'
        if 'mp_cores' in dir(cfg):
            ncores = cfg.mp_cores
        else:
            ncores = -2
        if ncores < 0:
            prefs.devices.cpp_standalone.openmp_threads = mp.cpu_count() + ncores
        elif ncores > 0:
            prefs.devices.cpp_standalone.openmp_threads = ncores
        runit = run_cpu
    device_args = dict(directory=working_dir)
    os.makedirs(working_dir, exist_ok=True)
    set_device(dev, **device_args)
    return functools.partial(runit, cfg, **device_args), working_dir


if __name__ == '__main__':
    cfg = importlib.import_module('.'.join(sys.argv[1].split('.')[0].split('/')))
    rng = np.random.default_rng()
    start_at = cfg.start_at.copy() if 'start_at' in dir(cfg) else {}
    generate_networks(cfg, rng, start_at)

    runit, working_dir = set_run_func(cfg)
    raw_fbase = cfg.raw_fbase if hasattr(cfg, 'raw_fbase') else None

    Xstim, Ystim = spatial.create_stimulus_locations(cfg.params)

    # Set up input templates
    if start_at.get('newtempl', True):
        X, Y, W, D = spatial.create_weights(cfg.params, rng)
        Net = model.create_network(X, Y, Xstim, Ystim, W, D, cfg.params, reset_dt=inputs.get_episode_duration(cfg.params))
        templates = [readout.setup_run(Net, cfg.params, rng, cfg.stimuli, cfg.pairings) for _ in range(cfg.N_templates)]
        if (start_at.get('net', 0) != 0
                or start_at.get('STD', cfg.STDs[0]) != cfg.STDs[0]
                or start_at.get('TA', cfg.TAs[0]) != cfg.TAs[0]
                or start_at.get('isi', cfg.ISIs[0]) != cfg.ISIs[0]):
            templ = start_at.get('templ', 0)
            preset = readout.load_results(cfg.fname.format(templ=templ, net=0, STD=cfg.STDs[0], TA=cfg.TAs[0], isi=cfg.ISIs[0]))
            for key in preset.keys():
                if key not in templates[templ]:
                    preset.pop(key)
            templates[templ] = preset
    else:
        templates = [
            readout.load_results(
                cfg.fname.format(templ=templ, net=0, STD=cfg.STDs[0], TA=cfg.TAs[0], isi=cfg.ISIs[0]),
                template_only=True)
            for templ in range(cfg.N_templates)]

    for templ, template in enumerate(templates):
        if templ < start_at.get('templ', 0):
            continue
        else:
            start_at.pop('templ', 0)
        with_dynamics = templ < cfg.N_templates_with_dynamics
        for net in range(cfg.N_networks):
            if net < start_at.get('net', 0):
                continue
            else:
                start_at.pop('net', 0)
            try:
                res = dd.io.load(cfg.netfile.format(net=net))
            except Exception as e:
                print(e)
            X, Y, W, D = res['X']*meter, res['Y']*meter, res['W'], res['D']
            for STD in cfg.STDs:
                for TA in cfg.TAs:
                    if STD < start_at.get('STD', 0) or TA < start_at.get('TA', 0):
                        continue
                    else:
                        start_at.pop('STD', 0)
                        start_at.pop('TA', 0)
                    Tstart = time.time()
                    for iISI, isi in enumerate(cfg.ISIs):
                        if isi < start_at.get('isi', cfg.ISIs[0]):
                            continue
                        else:
                            start_at.pop('isi', 0)
                        mod_params = {**cfg.params, 'ISI': isi*ms,
                                      'tau_rec': (0*ms, cfg.params['tau_rec'])[STD],
                                      'th_ampl': (0*mV, cfg.params['th_ampl'])[TA]}
                        
                        rundata = runit(template, with_dynamics, STD, TA, mod_params, X, Y, Xstim, Ystim, W, D,
                                        raw_fbase=None if raw_fbase is None else raw_fbase.format(**locals()))

                        try:
                            readout.save_results(cfg.fname.format(**locals()), rundata)
                        except Exception as e:
                            print(e)
                        brian_cleanup(working_dir)

                    print(f'Completed ISI sweep (templ {templ}, net {net}, STD {STD}, TA {TA}) after {(time.time()-Tstart)/60:.1f} minutes.')