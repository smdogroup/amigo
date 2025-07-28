import sys
import os

if sys.platform == "win32":
    libopenblas_dll_dir = r"C:\libs\openblas\bin"
    os.add_dll_directory(libopenblas_dll_dir)

from .amigo import Vector, OptimizationProblem, SparseCholesky, OrderingType
from .component import Component
from .model import Model
from .optimizer import Optimizer
from .utils import tocsr, topetsc
from .unary_operations import *


def get_include():
    from .amigo import AMIGO_INCLUDE_PATH

    return AMIGO_INCLUDE_PATH


def get_a2d_include():
    from .amigo import A2D_INCLUDE_PATH

    return A2D_INCLUDE_PATH
