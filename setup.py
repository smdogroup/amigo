import os
import sys
import mpi4py
from setuptools import setup, Extension
from subprocess import check_output
import glob


use_openmp = "--with-openmp" in sys.argv
if use_openmp:
    sys.argv.remove("--with-openmp")

use_debug = "--with-debug" in sys.argv
if use_debug:
    sys.argv.remove("--with-debug")


def get_mpi_flags():
    # Windows-specific MPI handling
    if sys.platform == "win32":
        # Microsoft MPI SDK paths
        mpi_sdk_base = r"C:\Program Files (x86)\Microsoft SDKs\MPI"
        inc_dirs = [os.path.join(mpi_sdk_base, "Include")]
        lib_dirs = [os.path.join(mpi_sdk_base, "Lib", "x64")]
        libs = ["msmpi"]  # Microsoft MPI library name
        return inc_dirs, lib_dirs, libs
    else:
        # Unix/Linux/Mac systems - use mpicxx
        # Split the output from the mpicxx command
        args = check_output(["mpicxx", "-show"]).decode("utf-8").split()

        # Determine whether the output is an include/link/lib command
        inc_dirs, lib_dirs, libs = [], [], []
        for flag in args:
            if flag[:2] == "-I":
                inc_dirs.append(flag[2:])
            elif flag[:2] == "-L":
                lib_dirs.append(flag[2:])
            elif flag[:2] == "-l":
                libs.append(flag[2:])

        return inc_dirs, lib_dirs, libs


def get_extensions():
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    import pybind11

    link_args = []
    compile_args = []
    define_macros = []

    inc_dirs, lib_dirs, libs = get_mpi_flags()
    inc_dirs.append(mpi4py.get_include())

    inc_dirs.append("include")
    headers = glob.glob("include/*.h")

    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")

    # Try to detect the actual git directory location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    git_root = current_dir

    # Navigate up to find the git directory containing both a2d and amigo
    while git_root and not (
        os.path.exists(os.path.join(git_root, "a2d"))
        and os.path.exists(os.path.join(git_root, "amigo"))
    ):
        parent = os.path.dirname(git_root)
        if parent == git_root:  # reached filesystem root
            break
        git_root = parent

    # If we found the git root, use it; otherwise fall back to the old logic
    if git_root and os.path.exists(os.path.join(git_root, "a2d")):
        a2d_include = os.path.join(git_root, "a2d", "include")
        amigo_include = os.path.join(git_root, "amigo", "include")
    else:
        # Fallback to old logic
        a2d_include = os.path.join(home, "git", "a2d", "include")
        amigo_include = os.path.join(home, "git", "amigo", "include")

    # Add METIS if available (optional on Windows)
    metis_include = os.path.join(home, "git", "tacs", "extern", "metis", "include")
    metis_lib = os.path.join(home, "git", "tacs", "extern", "metis", "lib")

    metis_available = False
    if os.path.exists(metis_include) and os.path.exists(metis_lib):
        inc_dirs.append(metis_include)
        lib_dirs.append(metis_lib)
        libs.append("metis")
        metis_available = True

    # Escape backslashes for Windows paths in C++ string literals
    a2d_include_escaped = a2d_include.replace("\\", "\\\\")
    amigo_include_escaped = amigo_include.replace("\\", "\\\\")

    with open("include/amigo_include_paths.h", "w") as f:
        f.write(f"#ifndef AMIGO_INCLUDE_PATHS_H\n")
        f.write(f"#define AMIGO_INCLUDE_PATHS_H\n")
        f.write(f'#define A2D_INCLUDE_PATH "{a2d_include_escaped}"\n')
        f.write(f'#define AMIGO_INCLUDE_PATH "{amigo_include_escaped}"\n')
        f.write(f"#endif  // AMIGO_INCLUDE_PATHS_H\n")

    if sys.platform == "darwin":
        from distutils import sysconfig

        vars = sysconfig.get_config_vars()
        vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-dynamiclib")

    if sys.platform == "win32":
        compile_args += ["/std:c++17", "/permissive-"]
        openblas_root = r"C:\libs\openblas"
        inc_dirs += [os.path.join(openblas_root, "include")]
        lib_dirs += [os.path.join(openblas_root, "lib")]
        libs += ["openblas"]
    elif sys.platform == "darwin":
        compile_args += ["-std=c++17"]
        if metis_available:
            define_macros += [("AMIGO_USE_METIS", "1")]
    else:
        link_args += ["-lblas", "-llapack"]
        compile_args += ["-std=c++17"]
        if metis_available:
            define_macros += [("AMIGO_USE_METIS", "1")]

    if use_openmp:
        if sys.platform == "win32":
            compile_args += ["/openmp"]
        else:
            compile_args += ["-fopenmp"]
            link_args += ["-fopenmp"]
        define_macros += [("AMIGO_USE_OPENMP", "1")]

    if use_debug:
        if sys.platform == "win32":
            compile_args += ["/Od", "/Zi"]
        else:
            compile_args += ["-g", "-O0"]

    ext_modules = [
        Extension(
            "amigo.amigo",
            sources=["amigo/amigo.cpp"],
            depends=headers,
            include_dirs=inc_dirs,
            libraries=libs,
            library_dirs=lib_dirs,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=define_macros,
        )
    ]

    return ext_modules


def get_include_dirs():
    import pybind11

    # Use the same path detection logic as get_extensions()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    git_root = current_dir

    # Navigate up to find the git directory containing both a2d and amigo
    while git_root and not (
        os.path.exists(os.path.join(git_root, "a2d"))
        and os.path.exists(os.path.join(git_root, "amigo"))
    ):
        parent = os.path.dirname(git_root)
        if parent == git_root:  # reached filesystem root
            break
        git_root = parent

    # If we found the git root, use it; otherwise fall back to the old logic
    if git_root and os.path.exists(os.path.join(git_root, "a2d")):
        a2d_include = os.path.join(git_root, "a2d", "include")
    else:
        # Fallback to old logic
        home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
        a2d_include = os.path.join(home, "git", "a2d", "include")

    pybind11_include = pybind11.get_include()
    include_dirs = [pybind11_include, a2d_include]

    return include_dirs


setup(
    name="amigo",
    ext_modules=get_extensions(),
    include_dirs=get_include_dirs(),
)
