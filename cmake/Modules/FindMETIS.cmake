find_path(METIS_INCLUDE_DIR
  NAMES metis.h
  HINTS
    ${METIS_ROOT}
    $ENV{METIS_ROOT}
  PATH_SUFFIXES include
)

find_library(METIS_LIBRARY
  NAMES metis
  HINTS
    ${METIS_ROOT}
    $ENV{METIS_ROOT}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
  REQUIRED_VARS METIS_INCLUDE_DIR METIS_LIBRARY
)

if(METIS_FOUND AND NOT TARGET METIS::metis)
  add_library(METIS::metis UNKNOWN IMPORTED)
  set_target_properties(METIS::metis PROPERTIES
    IMPORTED_LOCATION "${METIS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${METIS_INCLUDE_DIR}"
  )
endif()