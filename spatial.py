from typing import Optional
import numpy as np
from brian2 import Quantity, meter


def generate_circle_locations(n, radius, rng):
    if isinstance(radius, Quantity):
        radius /= meter
    loc = np.zeros((n, 2))
    mask = np.ones(n, bool)
    while mask.sum() > 0:
        loc[mask] = rng.uniform(-radius, radius, (mask.sum(), 2))
        mask[mask] = np.sum(loc[mask]**2, axis=1) > radius**2
    return loc.T*meter


def get_distance(xPre, yPre, xPost, yPost):
    return np.sqrt((xPre - xPost)**2 + (yPre - yPost)**2)


def get_boxcar_connections(xPre, yPre, xPost, yPost, radius,
                           rng : Optional[np.random.Generator] = None,
                           probability=1,
                           outdegree=None):
    nPre, nPost = len(xPre), len(xPost)
    assert len(yPre) == nPre and len(yPost) == nPost
    iPre = np.repeat(np.arange(nPre), nPost)
    iPost = np.tile(np.arange(nPost), nPre)
    dist = get_distance(xPre[iPre], yPre[iPre], xPost[iPost], yPost[iPost])
    mask = (dist < radius) & (dist > 0)
    mask_2D = mask.reshape(nPre, nPost)  # a writable view
    assert probability >= 1 or outdegree is None, 'Specify either probability or outdegree, not both'
    if probability < 1:
        assert rng is not None, 'Provide an RNG for stochastic connectivity.'
        probe = rng.uniform(size=mask.shape)
        mask &= probe < probability
    elif outdegree is not None:
        assert rng is not None, 'Provide an RNG for stochastic connectivity.'
        for pre in range(nPre):
            candidates = np.flatnonzero(mask_2D[pre])
            if candidates.size > outdegree:
                nontargets = rng.choice(candidates, candidates.size - outdegree, replace=False, shuffle=False)
                mask_2D[pre, nontargets] = 0
    return iPre[mask], iPost[mask], dist[mask]


def create_weights(params, rng):
    X, Y = generate_circle_locations(params['N'], params['r_dish'], rng)
    iPre_exc, iPost_exc, distance_exc = get_boxcar_connections(
        X[:params['N_exc']], Y[:params['N_exc']],
        X, Y,
        radius=params['r_exc'], outdegree=params['outdeg_exc'], rng=rng
    )
    iPre_inh, iPost_inh, distance_inh = get_boxcar_connections(
        X[params['N_exc']:], Y[params['N_exc']:],
        X, Y,
        radius=params['r_inh'], outdegree=params['outdeg_inh'], rng=rng
    )
    
    distribution = params.get('weight_distribution', 'lognormal')
    if distribution == 'lognormal':
        W_exc = rng.lognormal(params['w_exc_mean'], params['w_exc_sigma'], iPre_exc.shape)
        W_inh = rng.lognormal(params['w_inh_mean'], params['w_inh_sigma'], iPre_inh.shape)
    elif distribution == 'singular':
        W_exc = params['w_exc_mean']
        W_inh = params['w_inh_mean']
    W = np.full((params['N'], params['N']), np.nan)
    W[iPre_exc, iPost_exc] = W_exc
    W[iPre_inh + params['N_exc'], iPost_inh] = W_inh

    D = np.full_like(W, np.nan)
    D[iPre_exc, iPost_exc] = distance_exc
    D[iPre_inh, iPost_inh] = distance_inh

    return X, Y, W, D


def create_stimulus_locations(params):
    theta = np.arange(params['N_stimuli']) / params['N_stimuli'] * 2*np.pi
    r = params['stim_distribution_radius']
    return r*np.cos(theta), r*np.sin(theta)

def get_stimulated(X, Y, Xstim, Ystim, params):
    idx = np.empty((params['N_stimuli'], params['neurons_per_stim']), dtype=int)
    for i in range(params['N_stimuli']):
        dist = get_distance(X, Y, Xstim[i], Ystim[i])
        sorted = np.argsort(dist)
        idx[i] = sorted[:params['neurons_per_stim']]
    return idx

def get_distance_to_stim(stimid, W, X, Y, params):
    Wb = W>0
    d = np.zeros(X.shape, int)
    Xstim, Ystim = create_stimulus_locations(params)
    d[get_stimulated(X, Y, Xstim, Ystim, params)[stimid]] = 1
    for i in range(2, 10):
        pre = (d == i-1) & (np.arange(len(d)) < params['N_exc'])
        post = np.unique(np.nonzero(Wb[pre])[1])
        for p in post:
            if d[p] == 0:
                d[p] = i
        if np.sum(d==0) == 0:
            break
    d -= 1
    return d
