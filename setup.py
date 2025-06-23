import os
import sys
from setuptools import setup, Extension
from subprocess import check_output
import glob


use_openmp = "--with-openmp" in sys.argv
if use_openmp:
    sys.argv.remove("--with-openmp")


def get_extensions():
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    import pybind11

    inc_dirs, lib_dirs, libs = [], [], []

    inc_dirs.append("include")
    headers = glob.glob("include/*.h")

    pybind11_include = pybind11.get_include()

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

    inc_dirs += [
        os.path.join(
            os.environ.get("HOME"), "git", "tacs", "extern", "metis", "include"
        )
    ]
    lib_dirs.append(
        os.path.join(os.environ.get("HOME"), "git", "tacs", "extern", "metis", "lib")
    )
    libs.append("metis")

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

    link_args = []
    compile_args = []
    define_macros = []
    if sys.platform == "win32":
        compile_args += ["/std:c++17", "/permissive-"]
    elif sys.platform == "darwin":
        compile_args += ["-std=c++17"]
        define_macros += [("AMIGO_USE_METIS", "1")]
    else:
        link_args += ["-lblas", "-llapack"]
        compile_args += ["-std=c++17"]
        define_macros += [("AMIGO_USE_METIS", "1")]

    if use_openmp:
        compile_args += ["-fopenmp"]
        link_args += ["-fopenmp"]
        define_macros += [("AMIGO_USE_OPENMP", "1")]

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
