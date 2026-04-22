import sys
import os

if sys.platform == "win32":
    # Add DLL directories for Windows dependencies
    dll_dirs = [
        os.path.join(sys.prefix, "Library", "bin"),  # conda env or venv
        r"C:\Program Files\Microsoft MPI\Bin",  # System MS-MPI
    ]
    # BLAS/LAPACK + MUMPS from mumps-build (OpenBLAS-based, no MKL)
    _mumps_build = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "mumps-build",
        "build",
    )
    if os.path.isdir(_mumps_build):
        dll_dirs.append(_mumps_build)
    # Also check MUMPS_DLL_DIR env var for custom locations
    _mumps_env = os.environ.get("MUMPS_DLL_DIR")
    if _mumps_env:
        dll_dirs.append(_mumps_env)
    for dll_dir in dll_dirs:
        if os.path.exists(dll_dir):
            os.add_dll_directory(dll_dir)

from .amigo import (
    Vector,
    OptimizationProblem,
    CSRMat,
    SparseCholesky,
    SparseLDL,
    SolverType,
    OrderingType,
    MemoryLocation,
)
from .component import Component
from .model import Model
from .diagnostics import Diagnostics
from .algorithm import *
from .utils import *
from .unary_operations import *
from .interfaces import (
    ExternalOpenMDAOComponent,
    ExplicitOpenMDAOPostOptComponent,
    AmigoIndepVarComp,
)
from .trajectory import *
