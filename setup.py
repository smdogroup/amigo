import os
import sys
from setuptools import setup, Extension
from subprocess import check_output
import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11


def get_global_dir(files):
    tacs_root = os.path.abspath(os.path.dirname(__file__))
    new = []
    for f in files:
        new.append(os.path.join(tacs_root, f))
    return new


# def get_mpi_flags():
#     # Split the output from the mpicxx command
#     args = check_output(["mpicxx", "-show"]).decode("utf-8").split()

#     # Determine whether the output is an include/link/lib command
#     inc_dirs, lib_dirs, libs = [], [], []
#     for flag in args:
#         if flag[:2] == "-I":
#             inc_dirs.append(flag[2:])
#         elif flag[:2] == "-L":
#             lib_dirs.append(flag[2:])
#         elif flag[:2] == "-l":
#             libs.append(flag[2:])

#     return inc_dirs, lib_dirs, libs


# inc_dirs, lib_dirs, libs = get_mpi_flags()

inc_dirs, lib_dirs, libs = [], [], []

inc_dirs.append("include")
headers = glob.glob("include/*.h")

# Add the numpy/mpi4py directories
# inc_dirs.append(mpi4py.get_include())

# Get the include for pybind11
pybind11_include = pybind11.get_include()

a2d_include = os.path.join(os.environ.get("HOME"), "git", "a2d", "include")

# metis_include = os.path.join(
#     os.environ.get("HOME"), "git", "tacs", "extern", "metis", "include"
# )
# lib_dirs.append(
#     os.path.join(os.environ.get("HOME"), "git", "tacs", "extern", "metis", "lib")
# )
# libs.append("metis")

if sys.platform == "darwin":
    from distutils import sysconfig

    vars = sysconfig.get_config_vars()
    vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-dynamiclib")

# Create the Extension
ext_modules = [
    Extension(
        "amigo.amigo",
        sources=["amigo/amigo.cpp"],
        depends=headers,
        include_dirs=inc_dirs,
        libraries=libs,
        library_dirs=lib_dirs,
        extra_compile_args=["-std=c++17"],
    )
]

setup(
    name="amigo",
    ext_modules=ext_modules,
    include_dirs=[pybind11_include, a2d_include],
)
