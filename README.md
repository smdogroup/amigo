# Amigo: A friendly library for MDO on HPC

Amigo is a python library that is designed for solving multidisciplinary analysis and optimization problems with high-performance computing resources through automatically generated c++ wrappers. 

All application code is written in python and automatically compiled to c++. Automatic differentiation is used throughout to evaluate first and second derivatives using A2D. Different backend implementations are used depending on the computational environment: Serial, OpenMP, MPI and CUDA implementations can be used. The user python code and the model construction is independent of the target backend.

Integration with other MDO libraries is key for flexibility. Amigo contains interfaces to inject OpenMDAO models into Amigo models using `amigo.ExternalComponent`. Alternatively, Amigo can be used as a sub-optimization OpenMDAO component with accurate post-optimality derivatives.

A tutorial on how to use amigo and documentation can be found here: [https://smdogroup.github.io/amigo/](https://smdogroup.github.io/amigo/).

## Local FEM problem builder

The local Streamlit app lets you upload an Amigo-compatible mesh, define the
solution, geometry, and data spaces, enter an integrand map and boundary
condition map, and provide Python functions for the integrands and custom
boundary-condition callbacks. It includes Poisson and 2D source-sink
potential-flow templates, supports potential and weak formulations, solves
with Amigo, and displays scalar solution fields in an interactive PyVista
viewer. Left-drag rotates, middle-drag or Shift+left-drag pans, and right-drag
or scroll zooms. The potential flow template also offers vector glyphs for the
potential gradient (velocity).
The **Mesh details** panel shows each domain's visibility setting in a
single-click popover, with controls to show or hide all domains.

Problem definitions can be downloaded as JSON and loaded later. Presets include
the mesh as compressed data by default, so they can be moved between machines.
Only load presets and Python function code you trust: the app executes the
provided functions locally.

The editors use the same structures as `amigo.fem.Problem`. For example, a
scalar Poisson potential can use:

```json
{
  "solution_space": {"soln": {"u": "H1"}},
  "geometry_space": {"geo": {"x,y": "H1"}},
  "data_space": {"data": {"Jz": "const"}},
  "data_values": {"data.Jz": 10.0},
  "integrand_map": {
    "air": {"target": ["SURFACE1"], "integrand": "potential"}
  },
  "bc_map": {
    "fixed_edges": {
      "type": "dirichlet",
      "target": ["LEFT", "RIGHT", "BOTTOM", "TOP"],
      "input": ["u"]
    }
  }
}
```

Mesh targets must match the element-set/domain names shown under **Mesh
details**. Each function-space component contains one field group and space;
group multiple fields with comma-separated names (for example, `"u,v"`).
Integrand map values refer to function names in the Python editor. Non-Dirichlet
boundary conditions can refer to a callback function defined there as well.
Click **Apply settings** before downloading a preset to include the latest
editor values.

With Pixi installed, create the app environment and build Amigo:

```
pixi install -e app
pixi run -e app install-amigo
pixi run -e app poisson-gui
```

The app opens at `http://127.0.0.1:8501` and is only bound to the local
machine. Supported mesh formats are Abaqus INP and Nastran BDF, NAS, and DAT.
The first solve compiles the generated Amigo model module. The default preset
is a Poisson potential problem; the sidebar can also generate a small sample
rectangular mesh. The source/sink template expects a 2D mesh whose domain names
match the configured integrand and boundary targets; edit the maps for other
mesh naming conventions.

The **CRM MITC shell** template accepts Nastran BDF/NAS/DAT meshes containing
quadrilateral shell elements and uses the mesh's `SPC`/`SPC1` cards as
zero-valued displacement/rotation constraints. The result panel reports the
number of imported constrained DOFs for each shell field. Nonzero enforced
SPC values are rejected rather than silently treated as zero. It supports the
six MITCShell fields (`u`, `v`, `w`,
`rx`, `ry`, `rz`), editable material/thickness parameters, Python traction
integrands, and named load cases. The default traction uses the constant
`load.fz` data field; for example, load cases can set it to `1000.0` and
`2000.0`. The result viewer can show the original mesh, a deformed mesh, or
both overlaid. Deformation applies the solved `u`, `v`, and `w` translations,
with a configurable scale factor for visual amplification; rotation fields
remain scalar colors only. Select the Amigo LDL/Cholesky or SciPy sparse solver
and configure the supported Amigo ordering and stabilization options in the
**CRM shell** tab. This app environment installs `pyNastran` for BDF parsing
and pins NumPy below 2 to satisfy its current dependency requirements.

## Installing Amigo using pixi

The quickest way to get a working build on Linux, macOS or Windows is [pixi](https://pixi.sh), which installs the compiler toolchain, MPI, BLAS/LAPACK, METIS and MUMPS into a project-local environment:

```
pixi install
pixi run install-amigo
```

[pixi](https://pixi.sh) creates a reproducible, project-local conda environment
from `pixi.toml` and `pixi.lock`. It handles the dependencies that are otherwise
awkward to install by hand — MPI, BLAS/LAPACK, METIS and MUMPS — on Linux,
macOS (Apple Silicon) and Windows.

Nothing is installed system-wide: the environment lives in `.pixi/envs/` inside
the repository, and `pixi.lock` pins exact package versions for every platform,
so a checkout builds the same way on every machine.

### Install pixi

```bash
# Linux / macOS
curl -fsSL https://pixi.sh/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

### Quick start

From the repository root:

```bash
pixi install           # create the environment (first run downloads packages)
pixi run install-amigo # clone a2d if needed, then build and install amigo
```

`pixi run install-amigo` performs an editable install (`pip install -e .`), so
Python changes take effect immediately; re-run it after changing C++ sources.

### Platform prerequisites

| Platform | What you need beforehand |
| --- | --- |
| `linux-64` | Nothing. The compiler (`cxx-compiler`), OpenMPI, OpenBLAS, METIS and MUMPS all come from conda-forge. |
| `osx-arm64` | Nothing. Clang, `llvm-openmp`, OpenMPI, OpenBLAS, METIS and MUMPS come from conda-forge. |
| `win-64` | **Visual Studio Build Tools with the C++ workload.** conda-forge cannot ship MSVC, so it must be installed system-wide. |

#### Installing MSVC on Windows

Amigo's Python extension has to be built with the same compiler and C runtime
as CPython itself, which on Windows means MSVC. conda-forge cannot redistribute
it, so it is the one dependency pixi cannot provide. You do **not** need the
full Visual Studio IDE — the free standalone Build Tools are enough.

Run this from PowerShell:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

What to expect:

- The `--override` string **must stay on one line.** If your terminal wraps it
  across lines, `--add` arrives without its workload and the installer exits
  with code 87 (`Installer failed with exit code: 87`).
- `Microsoft.VisualStudio.Workload.VCTools` is the C++ build-tools workload;
  `--includeRecommended` pulls in the MSVC toolchain and the Windows SDK, both
  of which CMake needs. Without the SDK you get a compiler that cannot link.
- It downloads roughly 2 GB and takes several minutes with no progress output
  because of `--quiet`. It is not hung — `--wait` keeps winget blocked until
  the install finishes.
- Windows may prompt for elevation. A reboot is not usually required
  (`--norestart`), but restart the terminal so the new tools are picked up.

Verify:

```powershell
winget list --id Microsoft.VisualStudio.2022.BuildTools
```

You do not need a "Developer Command Prompt" or any `vcvarsall.bat` setup —
CMake locates MSVC on its own through the Visual Studio generator, so
`pixi run install-amigo` works from an ordinary shell. A successful configure
prints something like:

```
-- The CXX compiler identification is MSVC 19.44.35228.0
-- Check for working CXX compiler: .../VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe
```

If you already have Visual Studio 2019 or 2022 Community/Professional with the
"Desktop development with C++" workload, that works too — nothing extra to
install.

### Environments

`pixi.toml` defines three environments. Select one with `-e`; the default
environment needs no flag.

| Environment | Command | Contents |
| --- | --- | --- |
| `default` | `pixi run <task>` | Everything needed to build, install and run amigo: compiler toolchain, CMake, MPI, BLAS/LAPACK, METIS, MUMPS, and amigo's Python dependencies. |
| `dev` | `pixi run -e dev <task>` | `default` plus pytest, smt, black, pre-commit and OpenMDAO. Use this to run the test suite or the OpenMDAO examples. |
| `cuda` | `pixi run -e cuda <task>` | `default` plus the dev tools, the CUDA 12 toolkit and cuDSS. **linux-64 only**, and requires an NVIDIA driver supporting CUDA 12. |

Examples:

```bash
pixi run -e dev test                     # run the full test suite
pixi run -e cuda install-amigo-cuda      # build with CUDA + cuDSS
pixi shell -e dev                        # drop into an interactive shell
```

`pixi shell` activates the environment in your current terminal, which is
useful for running ad-hoc scripts or pointing an IDE at
`.pixi/envs/<env>/python`.

### Tasks

| Task | What it does |
| --- | --- |
| `a2d` | Clones [smdogroup/a2d](https://github.com/smdogroup/a2d) into `../a2d` if it is not already there. `CMakeLists.txt` expects the headers at that path. Every install task depends on this. |
| `install-amigo` | Editable install with METIS enabled, CUDA disabled, and OpenMP on (off on Windows, matching CI). |
| `install-amigo-cuda` | `cuda` environment only: editable install with CUDA, cuDSS and `CMAKE_CUDA_ARCHITECTURES=native`. |
| `test` | Runs `pytest tests/ -v`, installing amigo first. Use with `-e dev`. |
| `lint` / `format` | `black --check .` / `black .` |
| `clean` | Removes build directories and `__pycache__`. |

Because tasks declare their dependencies, `pixi run test` will install amigo
(and clone a2d) automatically if that has not happened yet.

### What pixi handles for you

- **MPI** — OpenMPI on Linux/macOS, MS-MPI on Windows, with `mpi4py` built
  against it. mpi4py's headers are needed at build time by `amigo/amigo.cpp`.
- **BLAS / LAPACK** — OpenBLAS on Linux/macOS; MKL on Windows, which
  `CMakeLists.txt` auto-detects at `<sys.prefix>/Library/lib/mkl_rt.lib`.
- **METIS** — `METIS_ROOT` is exported into the environment, so
  `cmake/Modules/FindMETIS.cmake` picks it up without any `-D` flags.
- **MUMPS** — `mumps-seq` from conda-forge, with `MUMPS_LIB_DIR` pointing at
  the environment's library directory. This replaces the manual
  coin-or/ThirdParty-Mumps build described in the main README. Not available
  on Windows (see below).
- **CMake and the compiler at run time** — `model.build_module()` compiles
  generated C++ while your script runs, so CMake, Ninja, pybind11 and a
  compiler are part of the runtime environment, not just the build.
- **a2d headers** — fetched by the `a2d` task.

### Platform notes

**Windows: no MUMPS.** conda-forge has no reliable MUMPS build for Windows, so
the `mumps` solver option is unavailable — the same limitation as the Windows CI
job, which skips `tests/functional`. If you build coin-or/ThirdParty-Mumps
under MSYS2 as described in the main README, amigo's loader finds it in
`~/mumps-coinor/bin` or in `%CONDA_PREFIX%\Library\bin`.

**Windows: OpenMP is off.** `install-amigo` passes
`-DAMIGO_ENABLE_OPENMP=OFF` on win-64 to match CI. Remove that flag from the
`[target.win-64.tasks.install-amigo]` command if you want to try it.

**Windows: `scripts/pixi-activate.bat`.** conda-forge's `msmpi` package sets
`MSMPI_INC` and friends from `%PREFIX%`, a variable that only exists during a
conda *build*. Under pixi it expands to nothing, leaving
`MSMPI_INC=\Library\include`, and CMake's `FindMPI` then constructs a broken
`MPI::MPI_CXX` target that makes every `try_compile` fail. The activation
script re-derives those paths from `%CONDA_PREFIX%`. It has to be a script
rather than an `activation.env` entry, because pixi does not expand
`$CONDA_PREFIX` inside `activation.env` values on Windows.

**Paths are passed through the environment, not the command line.** Windows
paths contain backslashes, and CMake treats those as escape characters in
`-D` arguments, so the install tasks deliberately pass no paths.

Commit `pixi.toml` **and** `pixi.lock` — the lock file is what makes builds
reproducible. `.pixi/` is generated and should stay out of git.

### Troubleshooting

**`CMAKE_CXX_COMPILER not set` / `'nmake' '-?' failed`** (Windows) — the VS
Build Tools are missing or lack the C++ workload. See
[Installing MSVC on Windows](#installing-msvc-on-windows), then re-run
`pixi run install-amigo`.

**`Imported target "MPI::MPI_CXX" includes non-existent path "/Library/include"`**
(Windows) — the activation script did not run. Confirm that
`scripts/pixi-activate.bat` exists and that `pixi run python -c "import os;
print(os.environ['MSMPI_INC'])"` prints an absolute path.

**`METIS not found: disabling METIS support`** — the pixi environment exports
`METIS_ROOT`, so this should not happen; a configure in the pixi environment
prints `METIS found: enabling AMIGO_USE_METIS`. If you see it anyway, check
that `pixi run python -c "import os; print(os.environ['METIS_ROOT'])"` prints
an absolute path, and that a stale CMake cache is not holding empty
`METIS_INCLUDE_DIR` / `METIS_LIBRARY` values — `pixi run clean` clears it.
amigo builds and runs correctly without METIS; the effect is on ordering
performance for large problems.

**Stale build after changing C++ sources** — run `pixi run clean` followed by
`pixi run install-amigo`.


## Building Amigo from scatch

Amigo uses CMake and scikit-build to build the primary amigo module and all model modules that comprise a multidisciplinary model.

To build and install the primary amigo module and its python wrappers, you can build the module with

```
pip install -e .
```

By default the OpenMP and CUDA parallelization are turned off. You can turn on or off these modules with additional command line arguments to pip. For an OpenMP and MPI install, use the pip command

```
pip install -e . -v \
    -Ccmake.args="-DCMAKE_CXX_COMPILER=mpicxx" \
    -Ccmake.args="-DAMIGO_ENABLE_OPENMP=ON" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDA=OFF"
```

For CUDA, we recommend using the NVIDIA CUDSS library that amigo can use to solve linear systems. Locally installing CUDSS is straightforward. It can be downloaded from NVIDIA webpage. Add the `$CUDSS_HOME/lib` and `$CUDSS_HOME/lib64` to your `LD_LIBRARY_PATH`. 

To enable CUDA and CUDSS and disable OpenMP, use the pip command

```
pip install -e . -v \
    -Ccmake.args="-DCMAKE_CXX_COMPILER=mpicxx" \
    -Ccmake.args="-DAMIGO_ENABLE_OPENMP=OFF" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDA=ON" \
    -Ccmake.args="-DCUDSS_HOME=/path/to/cudss" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDSS=ON" \
    -Ccmake.args="-DCMAKE_CUDA_ARCHITECTURES=native"
```

CUDA and OpenMP can be enabled at the same time. All component computations will utilize the GPU and the native Amigo solver will use OpenMP.

Amigo model modules inherit the build options that are selected during the install phase.

Installing Metis is recommended for good performance. Metis can be enabled through the command line via

```
pip install -e . -v \
    -Ccmake.args="-DCMAKE_CXX_COMPILER=mpicxx" \
    -Ccmake.args="-DAMIGO_ENABLE_OPENMP=ON" \
    -Ccmake.args="-DAMIGO_ENABLE_CUDA=OFF" \
    -Ccmake.args="-DAMIGO_ENABLE_METIS=ON" \
    -Ccmake.define.METIS_INCLUDE_DIR=/path/to/metis/include/ \
    -Ccmake.define.METIS_LIBRARY=/path/to/metis/compiled/library/libmetis.a
```

## MUMPS sparse solver

Amigo's interior-point optimizer can use [MUMPS](https://mumps-solver.org/) to obtain symmetric indefinite factorization of the KKT system. MUMPS is loaded at runtime, so it is not needed at build time. We use [coin-or/ThirdParty-Mumps](https://github.com/coin-or-tools/ThirdParty-Mumps) to build from source on all platforms, installing to `$HOME/mumps-coinor` so the Amigo loader finds the library automatically without any extra environment variables.

If you use pixi, `mumps-seq` is already part of the environment and the steps below are unnecessary on Linux and macOS.

### Linux

```bash
sudo apt install build-essential cmake gfortran libopenmpi-dev libopenblas-dev liblapack-dev libmetis-dev
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git ~/ThirdParty-Mumps
cd ~/ThirdParty-Mumps
./get.Mumps
./configure --prefix=$HOME/mumps-coinor
make -j$(nproc)
make install
```

### macOS

```bash
brew install cmake gcc open-mpi openblas
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git ~/ThirdParty-Mumps
cd ~/ThirdParty-Mumps
./get.Mumps
./configure --prefix=$HOME/mumps-coinor FC=gfortran F77=gfortran
make -j$(sysctl -n hw.ncpu)
make install
```

### Windows

Build via [MSYS2](https://www.msys2.org/) (UCRT64 shell):

```bash
pacman -S mingw-w64-ucrt-x86_64-{gcc-fortran,gcc,openblas,metis,scotch} make autoconf automake libtool
git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && cd ThirdParty-Mumps
./get.Mumps && ./configure --prefix=/ucrt64 && make -j$(nproc) && make install
```

Then add `C:\msys64\ucrt64\bin` to your Windows PATH so that `libdmumps.dll` is found at runtime.

## Rosenbrock example

Below are two short examples that illustrate some of the features of Amigo.

First, the Rosenbrock function is a frequently used example problem in optimization. In Amigo, all analysis occurs within classes that are derived from `amigo.Component`.

The inputs, constraints, outputs, objective function, data and class constants are defined in the constructor. Values are accessed through dictionary member data structures `self.inputs`, `self.constraints`, `self.outputs`, `self.objective`, `self.data` and `self.constants`.

The Rosenbrock component takes two inputs `x1` and `x2` and provides the objective value `obj` and a constraint `con`.

```python
import amigo as am

class Rosenbrock(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x1", value=-1.0, lower=-2.0, upper=2.0)
        self.add_input("x2", value=-1.0, lower=-2.0, upper=2.0)
        self.add_objective("obj")
        self.add_constraint("con", value=0.0, lower=-float("inf"), upper=0.0)

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        self.objective["obj"] = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
        self.constraints["con"] = x1**2 + x2**2 - 1.0

model = am.Model("rosenbrock")
model.add_component("rosenbrock", 1, Rosenbrock())

model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize()
```

The Amigo model is created by initializing the `amigo.Model` object and adding components to it. The code is then created and compiled by `model.build_module()`.

Note when `model.build_module()` is called, the module is compiled. Whenever python code within the component class is changed, the module must be re-built. This must occur before initialization.

## Cart pole system example

The system dynamics are encoded within a `CartComponent` class that inherits from `amigo.Component`. In the constructor, you must specify what `constants`, `inputs`, `constraints` and `data` (not illustrated in this example) the component requires.

The only class member function that is used by Amigo is the `compute` function. The `inputs`, `constraints`, `constants` and `data` must be extracted from the dictionary-like member objects in the compute function. These are not numerical objects, but instead encode the mathematical operations that are reinterpreted to generate c++ code.

In some applications it can be simpler to create multiple versions of the compute function. In this case, you can set `amigo.Component.set_args()`. which takes a list of dictionaries of keyword arguments, which will provide the compute function the provided keyword arguments.

```python
import amigo as am

class CartComponent(am.Component):
    def __init__(self):
        super().__init__()

        # Set constant values (these are compiled into static constexpr values in c++)
        self.add_constant("g", value=9.81)
        self.add_constant("L", value=0.5)
        self.add_constant("m1", value=1.0)
        self.add_constant("m2", value=0.3)

        # Input values specify variables that are under the control of the optimizer
        self.add_input("x", lower=-50, upper=50, units="N", label="control")
        self.add_input("q", shape=(4), label="state")
        self.add_input("qdot", shape=(4), label="rate")

        # Constraints within the optimization problem
        self.add_constraint("res", shape=(4), lower=0.0, upper=0.0, label="residual")

        return

    def compute(self):
        # The compute functions take 
        g = self.constants["g"]
        L = self.constants["L"]
        m1 = self.constants["m1"]
        m2 = self.constants["m2"]

        # Extract the input objects
        x = self.inputs["x"]
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        # Compute intermediate variables
        sint = am.sin(q[1])
        cost = am.cos(q[1])

        # Compute the residual
        res = 4 * [None]
        res[0] = q[2] - qdot[0]
        res[1] = q[3] - qdot[1]
        res[2] = (m1 + m2 * sint**2) * qdot[2] - (
            L * m2 * sint * q[3] * q[3] * x + m2 * g * cost * sint
        )
        res[3] = L * (m1 + m2 * sint**2) * qdot[3] + (
            L * m2 * cost * sint * q[3] * q[3] + x * cost + (m1 + m2) * g * sint
        )

        # Set the output
        self.constraints["res"] = res

        return
```

To create a model, you create the components, and specify how many instances of that component are used within the component group.

```python
# Create instances of the component classes
cart = CartComponent()
ic = InitialConditions()
fc = FinalConditions()

# Specify the module name
module_name = "cart_pole"
model = am.Model(module_name)

# Add the component classes to the model
model.add_component("cart", num_time_steps + 1, cart)
model.add_component("ic", 1, ic)
model.add_component("fc", 1, fc)
```

Variables are linked between components through an explicit linking process. Indices within the model are linked with text-based linking arguments. The names provided to the linking command are scoped by `component_name.variable_name`. 

You can also add sub-models to the model by calling `model.sub_model("sub_model", sub_model)`. In this case the scope becomes `sub_model.component_name.variable_name`. Any links specified in the sub-model are added to the model.

Linking establishes that two `inputs` from different components are the same. Linking two `constraints` or two `outputs` together means that the sum of the two output values are used as a constraint. You cannot link between types for instance linking `inputs` to `constraints` or `outputs`. 

```python
# Link the initial and final conditions
model.link("cart.q[0, :]", "ic.q[0, :]")
model.link(f"cart.q[{num_time_steps}, :]", "fc.q[0, :]")

# After variables are all linked, initialize the model
model.initialize()
```

Source and target indices can be supplied for more general linking relationships.
