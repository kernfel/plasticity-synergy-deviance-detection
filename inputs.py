import numpy_ as np
from brian2 import second


def set_input_sequence(Net, sequence, params, offset=0*second):
    t = np.arange(len(sequence)) * params['ISI'] + params['settling_period'] + offset
    if hasattr(Net, 'input_sequence'):
        sequence = np.concatenate([Net.input_sequence, sequence])
        t = np.concatenate([Net.input_sequence_t, t])
    Net.input_sequence = sequence
    Net.input_sequence_t = t
    Net['Input'+Net.suffix].set_spikes(sequence, t)
    return t[-1] + params['ISI']


def create_oddball(Net, params, A, B, rng : np.random.Generator, offset=0*second):
    sequence = np.tile([A] * (params['sequence_length']-1) + [B], params['sequence_count'])
    if params.get('fully_random_oddball', False):
        rng.shuffle(sequence)
    if Net is None:
        return sequence
    else:
        return sequence, set_input_sequence(Net, sequence, params, offset=offset)


def create_MSC(Net, params, rng : np.random.Generator, offset=0*second):
    sequence = np.arange(params['N_stimuli'])
    if params.get('fully_random_msc', False):
        sequence = np.tile(sequence, params['sequence_count'])
    rng.shuffle(sequence)
    if not params.get('fully_random_msc', False):
        sequence = np.tile(sequence, params['sequence_count'])
    if Net is None:
        return sequence
    else:
        return sequence, set_input_sequence(Net, sequence, params, offset=offset)


def get_episode_duration(params):
    data_duration = params['ISI']*params['sequence_length']*params['sequence_count']
    return params['settling_period'] + data_duration
