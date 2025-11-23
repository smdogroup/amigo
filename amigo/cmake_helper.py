from __future__ import annotations

import sys
from pathlib import Path


def get_cmake_dir() -> Path:
    """
    Return the directory containing Amigo's CMake package files
    (AmigoConfig.cmake, AmigoAddModule.cmake, AmigoTargets.cmake).

    Works for both normal and editable installs.
    """
    # Try relative to the imported amigo package
    import amigo as _amigo

    pkg_dir = Path(_amigo.__file__).resolve().parent
    candidate = pkg_dir / "cmake" / "amigo"
    if candidate.is_dir():
        return candidate

    # If we're in an editable install, the imported package might be
    # the source tree, while the *installed* package (with cmake files)
    # lives somewhere else on sys.path. Look for that.
    for entry in map(Path, sys.path):
        cmake_dir = entry / "amigo" / "cmake" / "amigo"
        if cmake_dir.is_dir():
            return cmake_dir

    raise RuntimeError(
        "Could not locate Amigo's CMake directory. "
        "Checked the imported package and all sys.path entries for "
        "'amigo/cmake/Amigo'. Is amigo installed correctly?"
    )
