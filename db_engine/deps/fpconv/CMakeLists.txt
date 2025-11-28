project(fpconv)

add_library(fpconv STATIC fpconv_dtoa.c fpconv_dtoa.h fpconv_powers.h)

set_target_properties(fpconv PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED YES
)

add_definitions('-fPIC')

target_include_directories(fpconv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
