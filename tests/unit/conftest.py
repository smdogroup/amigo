"""
Stub out compiled extensions and heavy dependencies before any amigo.model
import occurs, so that the pure-Python helpers can be imported without a
C++ build, MUMPS, or SMT installation.
"""

import sys
from unittest.mock import MagicMock

for mod in [
    "amigo.amigo",
    "mpi4py",
    "mpi4py.MPI",
    "pybind11",
    "scipy",
    "scipy.sparse",
    "scipy.sparse.linalg",
    "scipy.interpolate",
    "networkx",
]:
    sys.modules.setdefault(mod, MagicMock())
