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
from .utils import *
from .unary_operations import *
from .interfaces import (
    ExternalOpenMDAOComponent,
    ExplicitOpenMDAOPostOptComponent,
    AmigoIndepVarComp,
)
