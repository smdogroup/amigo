cmake_minimum_required(VERSION 3.22)
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

# Your shared headers for C++/CUDA backends
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../a2d/include)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Ask Python for the exact extension suffix
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')"
  OUTPUT_VARIABLE PY_EXT_SUFFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# A helper function to define one generated example module
function(add_amigo_example MODULE_DIR MODULE_NAME)
  set(EX_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/${MODULE_DIR})

  # Generated outputs
  set(GEN_HDR ${EX_DIR}/${MODULE_NAME}.h)
  set(GEN_CPP ${EX_DIR}/${MODULE_NAME}.cpp)

  # add_custom_command(
  #   OUTPUT  ${GEN_HDR} ${GEN_CPP}
  #   # Ensure the gen dir exists (safe if already exists)
  #   COMMAND ${CMAKE_COMMAND} -E make_directory ${BLD_GEN_DIR}
  #   # Run the example script to generate C++/header
  #   COMMAND ${Python3_EXECUTABLE} ${EX_PY} --generate
  #   BYPRODUCTS ${GEN_HDR} ${GEN_CPP}
  #   WORKING_DIRECTORY ${EX_DIR}
  #   COMMENT "Generating ${MODULE_NAME}.h/.cpp from ${MODULE_NAME}.py"
  #   VERBATIM
  # )

  # Build the pybind11 module from generated C++
  pybind11_add_module(${MODULE_NAME} MODULE ${GEN_CPP})

  if(AMIGO_ENABLE_CUDA)
    set_source_files_properties(${GEN_CPP} PROPERTIES LANGUAGE CUDA)
  endif()

  target_compile_features(${MODULE_NAME} PRIVATE cxx_std_17)
  target_include_directories(${MODULE_NAME}
    PRIVATE
      ${BLD_GEN_DIR}                 # for generated header
      ${MPI4PY_INCLUDE_DIR}          # if using mpi4py C-API
  )

  target_compile_options(${MODULE_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g -O0 -DDEBUG>
)

  target_link_libraries(${MODULE_NAME} PRIVATE MPI::MPI_CXX)

  if(AMIGO_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(${MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(${MODULE_NAME} PRIVATE AMIGO_USE_OPENMP=1)
  endif()

  if(AMIGO_ENABLE_CUDA)
    target_link_libraries(${MODULE_NAME} PRIVATE CUDA::cudart)
    target_compile_definitions(${MODULE_NAME} PRIVATE AMIGO_USE_CUDA=1)
    # Only needed if you compile .cu files for this target; harmless otherwise
    set_target_properties(${MODULE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  if(AMIGO_USE_MPI4PY_CAPI)
    target_compile_definitions(${MODULE_NAME} PRIVATE AMIGO_USE_MPI4PY_CAPI=1)
  endif()

  # Proper Python extension naming and no 'lib' prefix
  set_target_properties(${MODULE_NAME} PROPERTIES
    PREFIX ""
    SUFFIX "${PY_EXT_SUFFIX}"
  )

  # Put built extension in a neat per-example binary dir
  set(LOCAL_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/examples/${MODULE_NAME})
  set_target_properties(${MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY  ${LOCAL_OUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY  ${LOCAL_OUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY  ${LOCAL_OUT_DIR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )

  # Copy next to the example's Python driver so `import MODULE_NAME` works in-place
  add_custom_command(TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${MODULE_NAME}>" "${EX_DIR}/"
    COMMENT "Copying ${MODULE_NAME} extension next to ${MODULE_NAME}.py"
  )
endfunction()

# === Add your examples here ===
add_amigo_example(nozzle nozzle_module)
add_amigo_example(cart cart_pole)

# add_amigo_example(other_example_name)
