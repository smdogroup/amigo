import sys
import os

if sys.platform == "win32":
    # Add DLL directories for Windows dependencies
    dll_dirs = [
        r"C:\libs\openblas\bin",  # OpenBLAS
        r"C:\Program Files\Microsoft MPI\Bin",  # MS-MPI runtime
        os.path.join(sys.prefix, "Library", "bin"),  # MKL/BLAS in venv
    ]
    for dll_dir in dll_dirs:
        if os.path.exists(dll_dir):
            os.add_dll_directory(dll_dir)

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
