# Creates a Python extension module that uses the Amigo optimization framework.
# This function handles all the necessary setup including:
#   - Windows-specific dependency detection (MKL, MS-MPI)
#   - MSVC compiler configuration for C++17 conformance
#   - Output directory configuration for multi-config generators
#   - Linking to Amigo headers and optional CUDA backend
#
# Usage:
#   amigo_add_python_module(NAME module_name SOURCES source1.cpp [source2.cpp ...])
function(amigo_add_python_module)

  cmake_parse_arguments(AMIGO
    ""
    "NAME"
    "SOURCES"
    ${ARGN}
  )

  if(NOT AMIGO_NAME)
    message(FATAL_ERROR "amigo_add_python_module: NAME is required")
  endif()
  if(NOT AMIGO_SOURCES)
    message(FATAL_ERROR "amigo_add_python_module: SOURCES is required")
  endif()

  if(AMIGO_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 80)
    endif()
  endif()

  if(AMIGO_ENABLE_OPENMP)
    find_package(OpenMP)
  endif()

  find_package(amigo REQUIRED CONFIG)
  find_package(Python3 REQUIRED COMPONENTS Development.Module)
  find_package(pybind11 REQUIRED CONFIG)
  
  # Windows-specific dependency detection
  if(WIN32)
    # Auto-detect Intel MKL in Python venv for BLAS/LAPACK
    if(NOT DEFINED BLAS_LIBRARIES OR NOT DEFINED LAPACK_LIBRARIES)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os, sys; print(os.path.join(sys.prefix, 'Library'))"
        OUTPUT_VARIABLE VENV_LIBRARY_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE VENV_RESULT
      )
      
      if(VENV_RESULT EQUAL 0 AND EXISTS "${VENV_LIBRARY_PATH}/lib/mkl_rt.lib")
        message(STATUS "Found MKL in venv: ${VENV_LIBRARY_PATH}")
        set(BLAS_LIBRARIES "${VENV_LIBRARY_PATH}/lib/mkl_rt.lib" CACHE FILEPATH "BLAS library")
        set(LAPACK_LIBRARIES "${VENV_LIBRARY_PATH}/lib/mkl_rt.lib" CACHE FILEPATH "LAPACK library")
        list(APPEND CMAKE_PREFIX_PATH "${VENV_LIBRARY_PATH}")
        set(BLA_VENDOR Intel10_64lp)
      endif()
    endif()
    
    # Auto-detect Microsoft MPI SDK
    if(NOT DEFINED MPI_CXX_INCLUDE_PATH OR NOT DEFINED MPI_CXX_LIBRARIES)
      set(MSMPI_SDK_PATH "C:/Program Files (x86)/Microsoft SDKs/MPI")
      if(EXISTS "${MSMPI_SDK_PATH}/Include/mpi.h")
        message(STATUS "Found MS-MPI SDK: ${MSMPI_SDK_PATH}")
        set(MPI_CXX_INCLUDE_PATH "${MSMPI_SDK_PATH}/Include" CACHE PATH "MPI include path")
        set(MPI_CXX_LIBRARIES "${MSMPI_SDK_PATH}/Lib/x64/msmpi.lib" CACHE FILEPATH "MPI library")
      endif()
    endif()
  endif()
  
  # Find required dependencies
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)

  if (AMIGO_ENABLE_CUDSS)
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(CUDSS DEFAULT_MSG
      CUDSS_INCLUDE_DIR CUDSS_LIBRARY)

    if(CUDSS_FOUND)
      add_library(cudss::cudss UNKNOWN IMPORTED)
      set_target_properties(cudss::cudss PROPERTIES
        IMPORTED_LOCATION "${CUDSS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}"
      )
    endif()
  endif()

  pybind11_add_module(${AMIGO_NAME} MODULE ${AMIGO_SOURCES})
  set_target_properties(${AMIGO_NAME} PROPERTIES
    PREFIX ""
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
  
  # MSVC-specific settings
  if(MSVC)
    # Enable alternative tokens (and, or, not, etc.) for conforming C++ preprocessor
    target_compile_options(${AMIGO_NAME} PRIVATE /permissive- /Zc:preprocessor)
  endif()
  
  # Set output directories for all configurations
  set_target_properties(${AMIGO_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  
  # For multi-config generators (Visual Studio, Xcode), also set per-config directories
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set_target_properties(${AMIGO_NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach()

  # Always use headers
  target_link_libraries(${AMIGO_NAME} PRIVATE amigo::headers)

  # If the CUDA backend was built for this amigo install, link to it too
  if(TARGET amigo::backend)
    target_link_libraries(${AMIGO_NAME} PRIVATE amigo::backend)
  endif()

  if(AMIGO_ENABLE_CUDA)
    set_source_files_properties(${AMIGO_NAME} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()
