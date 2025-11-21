import sys
import os

if sys.platform == "win32":
    libopenblas_dll_dir = r"C:\libs\openblas\bin"
    os.add_dll_directory(libopenblas_dll_dir)

from .amigo import (
    Vector,
    OptimizationProblem,
    SparseCholesky,
    OrderingType,
    MemoryLocation,
)
from .component import Component
from .model import Model
from .optimizer import *
from .utils import tocsr, topetsc, BSplineInterpolant
from .unary_operations import *
from .interfaces import (
    ExternalOpenMDAOComponent,
    ExplicitOpenMDAOPostOptComponent,
    AmigoIndepVarComp,
)


def get_include():
    from .amigo import AMIGO_INCLUDE_PATH

    return AMIGO_INCLUDE_PATH


def get_a2d_include():
    from .amigo import A2D_INCLUDE_PATH

    return A2D_INCLUDE_PATH
