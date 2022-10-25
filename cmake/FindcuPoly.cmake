# Try to find the cuPoly library
# CUPOLY_FOUND - system has CUPOLY lib
# CUPOLY_INCLUDE_DIR - the CUPOLY include directory
# CUPOLY_LIBRARIES - Libraries needed to use CUPOLY

if (CUPOLY_INCLUDE_DIR AND CUPOLY_LIBRARIES)
    # Already in cache, be silent
    set(CUPOLY_FIND_QUIETLY TRUE)
endif (CUPOLY_INCLUDE_DIR AND CUPOLY_LIBRARIES)

FIND_PATH(CUPOLY_INCLUDE_DIR NAMES cuPoly/settings.h )
FIND_LIBRARY(CUPOLY_LIBRARIES NAMES cupoly libcupolybfv )
MESSAGE(STATUS "CUPOLY libs: " ${CUPOLY_LIBRARIES})

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUPOLY DEFAULT_MSG CUPOLY_INCLUDE_DIR CUPOLY_LIBRARIES)