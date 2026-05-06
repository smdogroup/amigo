"""
Stub out compiled extensions and heavy dependencies before any amigo.model
import occurs, so that the pure-Python helpers can be imported without a
C++ build, MUMPS, or SMT installation.

Only stubs modules that are not already importable — this preserves the real
compiled extension for tests like test_ldl.py that require it.
"""

import sys
import importlib
from unittest.mock import MagicMock


def _is_importable(name):
    """Return True if the module can be imported for real."""
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


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
    if not _is_importable(mod):
        sys.modules.setdefault(mod, MagicMock())
