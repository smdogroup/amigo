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

  # ------------------------------------------------------------
  # Dependencies needed to build a pybind11 module in the consumer
  # project. We do not call find_package(amigo) here because the
  # function is only available after the package is already found.
  # ------------------------------------------------------------
  find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
  find_package(pybind11 REQUIRED CONFIG)

  # CUDA support, if the installed Amigo package was built with CUDA
  if(AMIGO_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND DEFINED AMIGO_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "${AMIGO_CUDA_ARCHITECTURES}")
    endif()
  endif()

  # Optional OpenMP if Amigo was built with it
  if(AMIGO_ENABLE_OPENMP)
    find_package(OpenMP)
  endif()

  # These are already required transitively by amigo::headers, but
  # finding them explicitly here tends to make downstream diagnostics
  # clearer and keeps imported targets available in the consumer scope.
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)

  # ------------------------------------------------------------
  # Build the Python module
  # ------------------------------------------------------------
  pybind11_add_module(${AMIGO_NAME} MODULE ${AMIGO_SOURCES})

  set_target_properties(${AMIGO_NAME} PROPERTIES
    PREFIX ""
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  # For multi-config generators
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${CONFIG}" CONFIG_UPPER)
    set_target_properties(${AMIGO_NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach()

  # MSVC conformance flags
  if(MSVC)
    target_compile_options(${AMIGO_NAME} PRIVATE /permissive- /Zc:preprocessor)
  endif()

  # Link the exported Amigo interface target
  target_link_libraries(${AMIGO_NAME} PRIVATE amigo::headers)

  # If the installed package exported a CUDA backend, link it too
  if(TARGET amigo::backend)
    target_link_libraries(${AMIGO_NAME} PRIVATE amigo::backend)
  endif()

  # If OpenMP is enabled and available, it is usually already propagated
  # transitively by amigo::headers, but linking again here is harmless.
  if(AMIGO_ENABLE_OPENMP AND TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(${AMIGO_NAME} PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(AMIGO_ENABLE_CUDA)
    set_source_files_properties(${AMIGO_NAME} PROPERTIES LANGUAGE CUDA)
  endif()
endfunction()