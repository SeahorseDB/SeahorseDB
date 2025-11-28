project(hdr_histogram)

add_library(hdr_histogram STATIC hdr_histogram.c hdr_histogram.h hdr_atomic.h hdr_redis_malloc.h hdr_tests.h)

set_target_properties(hdr_histogram PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED YES
)

add_definitions('-DHDR_MALLOC_INCLUDE="hdr_redis_malloc.h"')
add_definitions('-fPIC')

target_include_directories(hdr_histogram PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
