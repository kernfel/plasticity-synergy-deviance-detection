import sys, os
from collections import defaultdict
from functools import wraps
import brian2.numpy_ as np
from brian2.units.fundamentalunits import (
    Quantity, fail_for_dimension_mismatch, is_dimensionless, DIMENSIONLESS)
from brian2.core.variables import VariableView


@wraps(np.concatenate)
def concatenate(arrays, /, **kwargs):
    if len(arrays) > 1:
        for array in arrays[1:]:
            fail_for_dimension_mismatch(
                arrays[0], array, 'All arguments must have the same units')
    elif len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate(arrays, **kwargs)
    if is_dimensionless(arrays[0]):
        return np.concatenate(arrays, **kwargs)
    else:
        dimensionless_arrays = [np.asarray(array) for array in arrays]
        return Quantity(
            np.concatenate(dimensionless_arrays, **kwargs),
            dim=arrays[0].dim, copy=False)


def ensure_unit(value, unit):
    if isinstance(value, dict):
        if isinstance(unit, dict):
            return {key: ensure_unit(val, unit[key]) if key in unit else val
                    for key, val in value.items()}
        else:
            return {key: ensure_unit(val, unit) for key, val in value.items()}
    elif isinstance(value, Quantity):
        # value must already be in units [unit]
        assert not isinstance(value/unit, Quantity)
    elif isinstance(value, VariableView):
        assert not isinstance(value[0]/unit, Quantity)
    elif type(value) in (list, tuple):
        return type(value)([ensure_unit(v, unit) for v in value])
    else:
        value = value * unit
    return value


def brian_cleanup(path='output'):
    for fname in os.listdir(f'{path}/results'):
        if fname.startswith('_'):
            os.remove(f'{path}/results/{fname}')


class Tree(defaultdict):
    def __init__(self, *args):
        super().__init__(*(args or (Tree,)))
    
    def asdict(self):
        return {k: v.asdict() if isinstance(v, Tree) else v for k, v in self.items()}


def isiterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True
