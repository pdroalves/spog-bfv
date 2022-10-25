# Try to find the GMP librairies
# GMP_FOUND - system has GMP lib
# GMP_INCLUDE_DIR - the GMP include directory
# GMP_LIBRARIES - Libraries needed to use GMP

if (GMP_INCLUDE_DIR AND GMP_LIBRARIES)
		# Already in cache, be silent
		set(GMP_FIND_QUIETLY TRUE)
endif (GMP_INCLUDE_DIR AND GMP_LIBRARIES)

find_path(GMP_INCLUDE_DIR NAMES gmp.h )
find_library(GMP_LIBRARIES NAMES gmp libgmp )
find_library(GMPXX_LIBRARIES NAMES gmpxx libgmpxx )

# Since the GMP version macros may be in a file included by gmp.h of the form
# gmp-.*[_]?.*.h (e.g., gmp-x86_64.h), we search each of them.
file(GLOB GMP_HEADERS "${GMP_INCLUDE_DIR}/gmp.h" "${GMP_INCLUDE_DIR}/gmp-*.h")
foreach(gmp_header_filename ${GMP_HEADERS})
	file(READ "${gmp_header_filename}" _gmp_version_header)
	string(REGEX MATCH
	  "define[ \t]+__GNU_MP_VERSION[ \t]+([0-9]+)" _gmp_major_version_match
	  "${_gmp_version_header}")
	if(_gmp_major_version_match)
	  set(GMP_MAJOR_VERSION "${CMAKE_MATCH_1}")
	  string(REGEX MATCH "define[ \t]+__GNU_MP_VERSION_MINOR[ \t]+([0-9]+)"
	    _gmp_minor_version_match "${_gmp_version_header}")
	  set(GMP_MINOR_VERSION "${CMAKE_MATCH_1}")
	  string(REGEX MATCH "define[ \t]+__GNU_MP_VERSION_PATCHLEVEL[ \t]+([0-9]+)"
	    _gmp_patchlevel_version_match "${_gmp_version_header}")
	  set(GMP_PATCHLEVEL_VERSION "${CMAKE_MATCH_1}")
	  set(GMP_VERSION
	    ${GMP_MAJOR_VERSION}.${GMP_MINOR_VERSION}.${GMP_PATCHLEVEL_VERSION})
	endif()
endforeach()

MESSAGE(STATUS "GMP found: " ${GMP_LIBRARIES} " version " ${GMP_VERSION})

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GMP DEFAULT_MSG GMP_INCLUDE_DIR GMP_LIBRARIES)

mark_as_advanced(GMP_INCLUDE_DIR GMP_LIBRARIES)

