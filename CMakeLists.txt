cmake_minimum_required(VERSION 3.25)
project(amigo LANGUAGES CXX)

# ------------------------
# User options
# ------------------------
option(AMIGO_ENABLE_CUDA "Build with CUDA (nvcc) support" OFF)
option(AMIGO_ENABLE_OPENMP "Build with OpenMP support" ON)

# If CUDA requested, activate the language (this enables nvcc toolchain)
if(AMIGO_ENABLE_CUDA)
enable_language(CUDA)
# Select architectures (adjust for your targets)
# e.g., 70 (V100), 80 (A100), 89 (Lovelace), 90 (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 80 90)
endif()
endif()

# Require a modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------
# Dependencies
# ------------------------
# Find Python to run a small query
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Query mpi4py's include dir from Python
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT EXISTS "${MPI4PY_INCLUDE_DIR}/mpi4py/mpi4py.h")
  message(FATAL_ERROR "mpi4py header not found; got: ${MPI4PY_INCLUDE_DIR}")
endif()

# Add to your targets that include <mpi4py/mpi4py.h>
target_include_directories(amigo PRIVATE "${MPI4PY_INCLUDE_DIR}")

# pybind11 will be provided via pip (pyproject "build-system" requires)
find_package(pybind11 CONFIG REQUIRED)

# OpenMP (optional)
if(AMIGO_ENABLE_OPENMP)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
message(WARNING "OpenMP requested but not found; continuing without OpenMP")
set(AMIGO_ENABLE_OPENMP OFF)
endif()
endif()

# ------------------------
# Core library
# ------------------------
add_library(amigo_core)
add_library(amigo::core ALIAS amigo_core)

# Core sources
target_sources(amigo_core PRIVATE
${CMAKE_CURRENT_SOURCE_DIR}/amigo/amigo.cpp
)

target_include_directories(amigo_core PUBLIC
${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(amigo_core PUBLIC
$<$<BOOL:${AMIGO_ENABLE_CUDA}>:AMIGO_USE_CUDA>
$<$<BOOL:${AMIGO_ENABLE_OPENMP}>:AMIGO_USE_OPENMP>
RUNTIME DESTINATION .)

#