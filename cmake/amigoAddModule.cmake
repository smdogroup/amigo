function(amigo_add_python_module)
  # Usage:
  #   amigo_add_python_module(NAME cart SOURCES cart.cpp)

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
  
  # Use BLAS/LAPACK/MPI settings from amigo installation instead of re-searching
  # These were found and cached during the main amigo build
  if(AMIGO_BLAS_LIBRARIES)
    set(BLAS_LIBRARIES "${AMIGO_BLAS_LIBRARIES}" CACHE FILEPATH "BLAS library" FORCE)
    set(BLAS_FOUND TRUE)
  else()
    find_package(BLAS REQUIRED)
  endif()
  
  if(AMIGO_LAPACK_LIBRARIES)
    set(LAPACK_LIBRARIES "${AMIGO_LAPACK_LIBRARIES}" CACHE FILEPATH "LAPACK library" FORCE)
    set(LAPACK_FOUND TRUE)
  else()
    find_package(LAPACK REQUIRED)
  endif()
  
  if(AMIGO_MPI_CXX_LIBRARIES AND AMIGO_MPI_CXX_INCLUDE_PATH)
    set(MPI_CXX_LIBRARIES "${AMIGO_MPI_CXX_LIBRARIES}" CACHE FILEPATH "MPI library" FORCE)
    set(MPI_CXX_INCLUDE_PATH "${AMIGO_MPI_CXX_INCLUDE_PATH}" CACHE PATH "MPI include" FORCE)
    set(MPI_CXX_FOUND TRUE)
    set(MPI_FOUND TRUE)
  else()
    find_package(MPI REQUIRED COMPONENTS CXX)
  endif()

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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  # Always use headers
  target_link_libraries(${AMIGO_NAME} PRIVATE amigo::headers)

  # If the CUDA backend was built for this amigo install, link to it too
  if(TARGET amigo::backend)
    target_link_libraries(${AMIGO_NAME} PRIVATE amigo::backend)
  endif()

  if(AMIGO_ENABLE_CUDA)
    set_source_files_properties(${AMIGO_NAME} PROPERTIES LANGUAGE CUDA)
  endif()

  # No 'lib' prefix
  set_target_properties(${AMIGO_NAME} PROPERTIES PREFIX "")

  # Put module right next to the generated .cpp / python file
  set_target_properties(${AMIGO_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
  
  # For multi-config generators (Visual Studio, Xcode), also set per-config directories
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set_target_properties(${AMIGO_NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endforeach()
endfunction()
