import sys
import os

if sys.platform == "win32":
    # Add venv Library\bin for MKL and other dependencies
    venv_lib_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.exists(venv_lib_bin):
        os.add_dll_directory(venv_lib_bin)

    # Add OpenBLAS DLL directory if specified or at default location
    openblas_dir = os.environ.get("OPENBLAS_DLL_DIR", r"C:\libs\openblas\bin")
    if os.path.exists(openblas_dir):
        os.add_dll_directory(openblas_dir)

    # Add MS-MPI bin directory (required for MPI support)
    msmpi_bin = os.environ.get("MSMPI_BIN", r"C:\Program Files\Microsoft MPI\Bin")
    if os.path.exists(msmpi_bin):
        os.add_dll_directory(msmpi_bin)

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
