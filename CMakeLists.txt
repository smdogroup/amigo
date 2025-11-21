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

# # Find METIS
# find_package(METIS REQUIRED)

# # Try to find METIS
# find_path(METIS_INCLUDE_DIR metis.h
#   HINTS
#     ${METIS_ROOT}/include
#     /usr/local/include
#     /usr/include
# )

# find_library(METIS_LIBRARY metis
#   HINTS
#     ${METIS_ROOT}/lib
#     /usr/local/lib
#     /usr/lib
# )

# if (NOT METIS_INCLUDE_DIR OR NOT METIS_LIBRARY)
#   message(FATAL_ERROR "Could not find METIS. Set METIS_ROOT or add its paths manually.")
# endif()

# # Create an imported target
# add_library(METIS::metis UNKNOWN IMPORTED)

# set_target_properties(METIS::metis PROPERTIES
#   IMPORTED_LOCATION "${METIS_LIBRARY}"
#   INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
# )

# target_link_libraries(amigo PRIVATE METIS::metis)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

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

target_compile_options(amigo PRIVATE
    $<$<CONFIG:Debug>:-g -O0 -DDEBUG>
)

# Link the CUDA runtime (shared)
if(AMIGO_ENABLE_CUDA)
  add_library(amigo_backend 
    src/csr_factor_cuda.cu
    src/csr_matrix_backend.cu
    src/optimizer_backend.cu
    src/vector_distribute.cu)
  target_link_libraries(amigo_backend PUBLIC CUDA::cudart CUDA::cusolver CUDA::cusparse MPI::MPI_CXX)

  target_include_directories(amigo_backend
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../a2d/include
  )

  # Feature flags visible to dependents
  target_compile_definitions(amigo_backend PUBLIC
    $<$<BOOL:${AMIGO_ENABLE_CUDA}>:AMIGO_USE_CUDA>
    $<$<BOOL:${AMIGO_ENABLE_OPENMP}>:AMIGO_USE_OPENMP>
  )

  set_target_properties(amigo_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_link_libraries(amigo PRIVATE amigo_backend CUDA::cudart CUDA::cublas
        CUDA::cublasLt CUDA::cusolver CUDA::cusparse)
  # CUDA::cudss
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
