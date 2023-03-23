"""
A dummy package to allow importing numpy and the unit-aware replacements of
numpy functions without having to know which functions are overwritten.

This can be used for example as ``import brian2.numpy_ as np``
"""

from brian2.numpy_ import *
from util import concatenate

# These will not be imported with a wildcard import to not overwrite the
# builtin names (mimicking the numpy behaviour)
from builtins import bool, int, float, complex, object, bytes, str

from numpy.core import round, abs, max, min

import brian2.numpy_
__all__ = []
__all__.extend(brian2.numpy_.__all__)
__all__.extend('concatenate')
