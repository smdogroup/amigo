cmake_minimum_required(VERSION 3.25)
project(amigo LANGUAGES CXX)

# ---------- Options ----------
option(AMIGO_ENABLE_CUDA   "Build with CUDA (nvcc) support" OFF)
option(AMIGO_ENABLE_OPENMP "Build with OpenMP support"     ON)

# ---------- Toolchains / languages ----------
if(AMIGO_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------- Dependencies ----------
# Ask CMake to locate BLAS and LAPACK.
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# MPI (C++ API) for code that uses MPI
find_package(MPI REQUIRED COMPONENTS CXX)

# OpenMP (optional)
if(AMIGO_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
  else()
    message(WARNING "OpenMP requested but not found; disabling")
    set(AMIGO_ENABLE_OPENMP OFF)
  endif()
endif()

# Python / pybind11 for the extension module
find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
find_package(pybind11 CONFIG REQUIRED)

# Find mpi4py header file
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import mpi4py, sys; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE MPI4PY_RC
)
if(NOT MPI4PY_RC EQUAL 0 OR NOT EXISTS "${MPI4PY_INCLUDE_DIR}/mpi4py/mpi4py.h")
  message(FATAL_ERROR "mpi4py header not found; got: ${MPI4PY_INCLUDE_DIR}")
endif()

# ---------- Header-only core ----------
# Carries include dirs and compile definitions, but no sources
add_library(amigo_headers INTERFACE)
add_library(amigo::headers ALIAS amigo_headers)

target_include_directories(amigo_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add headers for this target and anything that links to it
target_include_directories(amigo_headers
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../a2d/include
)

# Feature flags visible to dependents
target_compile_definitions(amigo_headers INTERFACE
  $<$<BOOL:${AMIGO_ENABLE_CUDA}>:AMIGO_USE_CUDA>
  $<$<BOOL:${AMIGO_ENABLE_OPENMP}>:AMIGO_USE_OPENMP>
)

# MPI / OpenMP linkage for any TU that includes headers and calls MPI/OpenMP
target_link_libraries(amigo_headers INTERFACE MPI::MPI_CXX)
if(AMIGO_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(amigo_headers INTERFACE OpenMP::OpenMP_CXX)
endif()

# ---------- Python extension module (pybind11) ----------
# This is your only .cpp file
pybind11_add_module(amigo MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/amigo/amigo.cpp
)

# Link the CUDA runtime (shared)
if(AMIGO_ENABLE_CUDA)
  target_link_libraries(amigo PRIVATE CUDA::cudart CUDA::cudss)
endif()

# Add LAPACK and BLAS
target_link_libraries(amigo PRIVATE LAPACK::LAPACK BLAS::BLAS)

# Link the header-only lib (brings MPI/OpenMP), and Python module lib
target_link_libraries(amigo PRIVATE
  amigo::headers
  Python3::Module     # pybind11 usually adds this, but keep explicit
)

target_include_directories(amigo PRIVATE "${MPI4PY_INCLUDE_DIR}")

# Install amigo
install(TARGETS amigo LIBRARY DESTINATION amigo)
