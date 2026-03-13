import sys
import os

if sys.platform == "win32":
    # Add DLL directories for Windows dependencies
    # Conda env / venv paths (preferred)
    dll_dirs = [
        os.path.join(sys.prefix, "Library", "bin"),  # conda env or venv (MKL, MPI)
    ]
    # Also check CONDA_PREFIX if it differs from sys.prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and os.path.normpath(conda_prefix) != os.path.normpath(sys.prefix):
        dll_dirs.append(os.path.join(conda_prefix, "Library", "bin"))
    # Fallback paths for system-wide installs
    dll_dirs.extend(
        [
            r"C:\Program Files\Microsoft MPI\Bin",  # System MS-MPI
            r"C:\libs\openblas\bin",  # Manual OpenBLAS install
        ]
    )
    for dll_dir in dll_dirs:
        if os.path.exists(dll_dir):
            os.add_dll_directory(dll_dir)

from .amigo import (
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
