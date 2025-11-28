project(linenoise)

add_library(linenoise STATIC linenoise.c linenoise.h)

set_target_properties(linenoise PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED YES
)

target_include_directories(linenoise PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
