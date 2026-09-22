"""Numba projection backend with compatibility module aliases."""

from . import nn as NN
from . import ns as NS
from . import sn as SN
from . import ss as SS
from . import tensor as TENSOR
from . import terminal as TERMINAL

__all__ = ["NN", "NS", "SN", "SS", "TENSOR", "TERMINAL"]
