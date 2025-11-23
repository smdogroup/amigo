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
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(MPI REQUIRED COMPONENTS CXX)

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
endfunction()
