cmake_minimum_required(VERSION 3.25)
project(amigo LANGUAGES CXX)

# Includes
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Set the amigo python package root under the install prefix.
# For scikit-build, CMAKE_INSTALL_PREFIX points at the "install" staging dir,
# and the wheel machinery will move this under site-packages/amigo/.
set(AMIGO_PY_PACKAGE_DIR "amigo")

# Package options
option(AMIGO_ENABLE_CUDA   "Build with CUDA (nvcc) support" OFF)
option(AMIGO_ENABLE_OPENMP "Build with OpenMP support"     ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Look for CUDA if enabled
if(AMIGO_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
endif()

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

# Amigo headers
add_library(headers INTERFACE)
add_library(amigo::headers ALIAS headers)

target_compile_definitions(headers INTERFACE
  $<$<BOOL:${AMIGO_ENABLE_CUDA}>:AMIGO_USE_CUDA>
  $<$<BOOL:${AMIGO_ENABLE_OPENMP}>:AMIGO_USE_OPENMP>
)

# MPI / OpenMP linkage for any TU that includes headers and calls MPI/OpenMP
target_link_libraries(headers INTERFACE MPI::MPI_CXX)
if(AMIGO_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(headers INTERFACE 
    OpenMP::OpenMP_CXX)
endif()

# Add LAPACK and BLAS
target_link_libraries(headers INTERFACE 
  LAPACK::LAPACK BLAS::BLAS)

# Add the pybind11 wrapper
pybind11_add_module(amigo MODULE
  ${CMAKE_CURRENT_SOURCE_DIR}/amigo/amigo.cpp
)

# Install amigo headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION "${AMIGO_PY_PACKAGE_DIR}/include"
)

# Install a2d headers alongside (adjust if you want a different layout)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../a2d/include/
    DESTINATION "${AMIGO_PY_PACKAGE_DIR}/a2d/include"
)

target_include_directories(headers INTERFACE
  # When building this project
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../a2d/include>

  # When consumed via find_package(Amigo)
  $<INSTALL_INTERFACE:amigo/include>
  $<INSTALL_INTERFACE:amigo/a2d/include>
)

# If Cuda is enabled, then we need to build the backend files
if(AMIGO_ENABLE_CUDA)
  add_library(backend 
    src/csr_factor_cuda.cu
    src/csr_matrix_backend.cu
    src/optimizer_backend.cu
    src/vector_distribute.cu)
  target_link_libraries(backend PUBLIC 
    CUDA::cudart CUDA::cublas CUDA::cublasLt 
    CUDA::cusolver CUDA::cusparse MPI::MPI_CXX)

  # Feature flags visible to dependents
  target_compile_definitions(backend PUBLIC
    $<$<BOOL:${AMIGO_ENABLE_CUDA}>:AMIGO_USE_CUDA>
    $<$<BOOL:${AMIGO_ENABLE_OPENMP}>:AMIGO_USE_OPENMP>
  )

  set_target_properties(backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_link_libraries(backend PUBLIC headers)

  target_link_libraries(amigo PRIVATE backend)
endif()

target_include_directories(amigo
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Link the header-only files to amigo
target_link_libraries(amigo PRIVATE
  amigo::headers Python3::Module)

target_include_directories(amigo PRIVATE 
  "${MPI4PY_INCLUDE_DIR}")

# Install the Python extension into the amigo Python package
set_target_properties(amigo PROPERTIES PREFIX "")

install(TARGETS amigo
  LIBRARY DESTINATION "${AMIGO_PY_PACKAGE_DIR}"
)

# Always export the header-only interface
install(TARGETS headers
  EXPORT amigoTargets 
  DESTINATION "${AMIGO_PY_PACKAGE_DIR}"
)

# If CUDA backend exists, export the CUDA backend
if(AMIGO_ENABLE_CUDA)
  add_library(amigo::backend ALIAS backend)
  install(TARGETS backend
    EXPORT amigoTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

# Export file with imported targets (amigo::headers, amigo::backend)
install(EXPORT amigoTargets
    NAMESPACE amigo::
    FILE amigoTargets.cmake
    DESTINATION "${AMIGO_PY_PACKAGE_DIR}/cmake/amigo"
)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/amigoConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/amigoConfig.cmake
  INSTALL_DESTINATION "${AMIGO_PY_PACKAGE_DIR}/cmake/amigo"
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/amigoConfig.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/amigoAddModule.cmake
  DESTINATION "${AMIGO_PY_PACKAGE_DIR}/cmake/amigo"
)