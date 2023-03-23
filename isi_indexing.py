import itertools
import numpy as np

from digest import conds
from util import isiterable


def pairlabel(cfg, ipair):
    return f'{cfg.pairings[ipair][0]}/{cfg.pairings[ipair][1]}'


def getindex(cfg, **kwargs):
    if 'isi' in kwargs:
        if isiterable(kwargs['isi']):
            iISI = [cfg.ISIs.index(isi) for isi in kwargs['isi']]
        else:
            iISI = cfg.ISIs.index(kwargs['isi'])
    else:
        iISI = kwargs.get('iISI', slice(None))
    if 'cond' in kwargs:
        if isiterable(kwargs['cond']) and not isinstance(kwargs['cond'], str):
            icond = [conds.index(cond) for cond in kwargs['cond']]
        else:
            icond = conds.index(kwargs['cond'])
    else:
        icond = kwargs.get('icond', slice(None))
    idx = (
        kwargs.get('templ', slice(None)),
        kwargs.get('net', slice(None)),
        kwargs.get('STD', slice(None)),
        kwargs.get('TA', slice(None)),
        iISI,
        kwargs.get('ipair', slice(None)),
        kwargs.get('istim', slice(None)),
        icond)
    iterable_idx = sum([isiterable(i) for i in idx])
    slice_idx = sum([isinstance(i, slice) for i in idx])
    assert iterable_idx==0 or slice_idx==0, 'Mixing slices and lists-of-indices is not supported.'
    # ... because the order of indexing is inconsistent in numpy. Example:
    #  np.arange(12).reshape(1,3,4)[0, :, (0,1,2,3)].shape  # Received: (4, 3) ; Expected: (3, 4)
    #  np.arange(12).reshape(1,3,4)[0, :, :].shape  # Received: (3, 4) as expected
    # See: https://github.com/numpy/numpy/pull/6256 https://github.com/numpy/numpy/pull/6075 
    return idx


def getaxis(tag : str | tuple | list, **kwargs):
    if type(tag) != str:
        return tuple([getaxis(t, **kwargs) for t in tag])
    axis = 0
    for i, itertag in enumerate((['templ'], ['net'], ['STD'], ['TA'], ['isi','iISI'], ['pair','ipair'], ['stim','istim'], ['cond','icond'])):
        if tag in itertag:
            return axis
        for itertag_ in itertag:
            if itertag_ in kwargs and not isiterable(kwargs[itertag_]):
                axis -= 1
                break
        axis += 1


def getlabels(cfg, ntempl, with_cond, **kwargs):
    title_chunks = []
    descriptor_chunks = []

    templ_ = kwargs.get('templ', range(ntempl))
    if isiterable(templ_):
        title_chunks.append([f'template {templ}' for templ in templ_])
    else:
        descriptor_chunks.append(f'template {templ_}')

    net_ = kwargs.get('net', range(cfg.N_networks))
    if isiterable(net_):
        title_chunks.append([f'net {net}' for net in net_])
    else:
        descriptor_chunks.append(f'net {net_}')

    STD_ = kwargs.get('STD', cfg.STDs)
    if isiterable(STD_):
        title_chunks.append([f'STD {STD}' for STD in STD_])
    else:
        descriptor_chunks.append(f'STD {STD_}')

    TA_ = kwargs.get('TA', cfg.TAs)
    if isiterable(TA_):
        title_chunks.append([f'TA {TA}' for TA in TA_])
    else:
        descriptor_chunks.append(f'TA {TA_}')

    isi_ = kwargs.get('isi', cfg.ISIs[kwargs.get('iISI', slice(None))])
    if isiterable(isi_):
        title_chunks.append([f'ISI {isi} ms' for isi in isi_])
    else:
        descriptor_chunks.append(f'ISI {isi_} ms')

    if 'ipair' not in kwargs or 'istim' not in kwargs:
        raise NotImplemented
    else:
        descriptor_chunks.append(f'pair {pairlabel(cfg, kwargs["ipair"])}')
        descriptor_chunks.append(f'stim {cfg.pairings[kwargs["ipair"]][kwargs["istim"]]}')

    if with_cond:
        cond_ = kwargs.get('cond', conds[kwargs.get('icond', slice(None))])
        if isiterable(cond_) and not isinstance(cond_, str):
            title_chunks.append(cond_)
        else:
            descriptor_chunks.append(cond_)

    return [', '.join(chunk) for chunk in itertools.product(*title_chunks)], ', '.join(descriptor_chunks)


def hist_view(cfg, hists, **kwargs):
    return hists[getindex(cfg, **kwargs)].reshape(-1, *hists.shape[-2:])


def hist_labels(cfg, hists, **kwargs):
    return getlabels(cfg, hists.shape[0], True, **kwargs)


def nspikes_labels(cfg, **kwargs):
    return getlabels(cfg, cfg.N_templates, False, **kwargs)


def get_onset_ordering(cfg, pspike, limit=None, filter=None, shift=False, tmax=None, **kwargs):
    if len(pspike.shape) > 3:
        pspike = hist_view(cfg, pspike, **kwargs)
    elif len(pspike.shape) == 2:
        pspike = pspike[None, :, :]
    hist_sum = pspike.sum(0)
    first_index = np.sum(np.cumsum(hist_sum, axis=1) == 0, axis=1)
    safe_first_index = first_index.copy()
    safe_first_index[first_index >= hist_sum.shape[1]] = 0
    first_intensity = hist_sum[np.arange(hist_sum.shape[0]), safe_first_index]
    
    onset_sort = np.lexsort((-first_intensity, first_index))
    order = onset_sort[limit] if type(limit) == slice else onset_sort[:limit]
    if filter is not None:
        order = order[np.isin(order, filter)]
    tmax = tmax or np.flatnonzero(hist_sum[order].sum(axis=0))[-1] + 1

    if shift:
        order = order[first_index[order] < tmax]
        tmax = min(tmax, pspike.shape[-1] - np.max(first_index[order]))
    
    index_N = np.repeat(order.reshape(-1,1), tmax, 1)
    index_t = np.repeat(np.arange(tmax).reshape(1,-1), len(order), 0)
    
    if shift:
        index_t += first_index[order].reshape(-1, 1)

    return index_N, index_t
