cmake_minimum_required(VERSION 3.13)
find_program(MAKE_EXE NAMES gmake nmake make)

project(vdb LANGUAGES CXX C)

# https://www.cmake.org/cmake/help/latest/policy/CMP0025.html
#
# Compiler id for Apple Clang is now AppleClang.
cmake_policy(SET CMP0025 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0042.html
#
# Enable MACOSX_RPATH by default. @rpath in a target's install name is
# a more flexible and powerful mechanism than @executable_path or
# @loader_path for locating shared libraries.
cmake_policy(SET CMP0042 NEW)

# https://www.cmake.org/cmake/help/latest/policy/CMP0054.html
#
# Only interpret if() arguments as variables or keywords when unquoted.
cmake_policy(SET CMP0054 NEW)

# https://www.cmake.org/cmake/help/latest/policy/CMP0057.html
#
# Support new if() IN_LIST operator.
cmake_policy(SET CMP0057 NEW)

# https://www.cmake.org/cmake/help/latest/policy/CMP0063.html
#
# Adapted from Apache Kudu: https://github.com/apache/kudu/commit/bd549e13743a51013585
# Honor visibility properties for all target types.
cmake_policy(SET CMP0063 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0068.html
#
# RPATH settings on macOS do not affect install_name.
cmake_policy(SET CMP0068 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0074.html
#
# find_package() uses <PackageName>_ROOT variables.
cmake_policy(SET CMP0074 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0091.html
#
# MSVC runtime library flags are selected by an abstraction.
cmake_policy(SET CMP0091 NEW)

# https://cmake.org/cmake/help/latest/policy/CMP0135.html
#
# CMP0135 is for solving re-building and re-downloading.
# We don't have a real problem with the OLD behavior for now
# but we use the NEW behavior explicitly to suppress CMP0135
# warnings.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

if(NOT DEFINED WITH_TESTS)
  set(WITH_TESTS "OFF")
endif()

if(NOT DEFINED RUN_TESTS)
  set(RUN_TESTS "OFF")
endif()

if(NOT DEFINED _REMOVE_ALL_AFTER_TEST)
  set(_REMOVE_ALL_AFTER_TEST "OFF")
endif()

if(NOT DEFINED APPLY_CLANG_FORMAT)
  option(APPLY_CLANG_FORMAT "execute clang-format before building seahorse-server" ON)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/output")

# Set default build type to Debug
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Set compiler flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()

# Set an IO option in SeahorseDB
if(NOT DEFINED USE_AIO)
  set(USE_AIO "SYNC")
endif()

set(PREV_USE_AIO "" CACHE STRING "Previous USE_AIO value")
string(TOUPPER "${USE_AIO}" USE_AIO_UPPER)

if(NOT "${USE_AIO}" STREQUAL "${PREV_USE_AIO}")
  message(STATUS "IO option changed from ${PREV_USE_AIO} to ${USE_AIO}, forcing rebuild")
  set(PREV_USE_AIO ${USE_AIO} CACHE STRING "Previous USE_AIO value" FORCE)
  file(REMOVE "${CMAKE_BINARY_DIR}/CMakeCache.txt")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -g -ggdb -D_GNU_SOURCE -Werror=implicit-function-declaration -march=native -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -ggdb --std=c++17 -march=native -Wno-missing-field-initializers")

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_C_STANDARD: ${CMAKE_C_STANDARD}")
message(STATUS "CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_C_COMPILER_")
message(STATUS " ID       \t: ${CMAKE_C_COMPILER_ID}")
message(STATUS " Version  \t: ${CMAKE_C_COMPILER_VERSION}")
message(STATUS " Path     \t: ${CMAKE_C_COMPILER}")
message(STATUS "CMAKE_CXX_COMPILER_")
message(STATUS " ID       \t: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS " Version  \t: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS " Path     \t: ${CMAKE_CXX_COMPILER}")
message(STATUS "WIN32 : ${WIN32}")
message(STATUS "APPLE : ${APPLE}")
message(STATUS "UNIX : ${UNIX}")
message(STATUS "CMAKE_SYSTEM_NAME : ${CMAKE_SYSTEM_NAME}")

include(deps/CMakeLists.txt)

# Create release header files(release.c/.h)
execute_process(COMMAND sh mkreleasehdr.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)

# List of source files for redis-server, derived from the object files mentioned
set(REDIS_SERVER_SOURCES
    src/acl.c
    src/adlist.c
    src/ae.c
    src/anet.c
    src/aof.c
    src/bio.c
    src/bitops.c
    src/blocked.c
    src/call_reply.c
    src/childinfo.c
    src/cluster.c
    src/commands.c
    src/config.c
    src/connection.c
    src/crc16.c
    src/crc64.c
    src/crcspeed.c
    src/db.c
    src/debug.c
    src/defrag.c
    src/dict.c
    src/endianconv.c
    src/eval.c
    src/evict.c
    src/expire.c
    src/function_lua.c
    src/functions.c
    src/geo.c
    src/geohash.c
    src/geohash_helper.c
    src/hyperloglog.c
    src/intset.c
    src/latency.c
    src/lazyfree.c
    src/listpack.c
    src/localtime.c
    src/logreqres.c
    src/lolwut.c
    src/lolwut5.c
    src/lolwut6.c
    src/lzf_c.c
    src/lzf_d.c
    src/memtest.c
    src/module.c
    src/monotonic.c
    src/mt19937-64.c
    src/multi.c
    src/networking.c
    src/notify.c
    src/object.c
    src/pqsort.c
    src/pubsub.c
    src/quicklist.c
    src/rand.c
    src/rax.c
    src/rdb.c
    src/redis-check-aof.c
    src/redis-check-rdb.c
    src/release.c
    src/release.c
    src/release.h
    src/replication.c
    src/resp_parser.c
    src/rio.c
    src/script.c
    src/script_lua.c
    src/sds.c
    src/sentinel.c
    src/server.c
    src/setcpuaffinity.c
    src/setproctitle.c
    src/sha1.c
    src/sha256.c
    src/siphash.c
    src/slowlog.c
    src/socket.c
    src/sort.c
    src/sparkline.c
    src/strl.c
    src/syncio.c
    src/syscheck.c
    src/t_hash.c
    src/t_list.c
    src/t_set.c
    src/t_stream.c
    src/t_string.c
    src/t_zset.c
    src/timeout.c
    src/tls.c
    src/tracking.c
    src/unix.c
    src/util.c
    src/ziplist.c
    src/zipmap.c
    src/zmalloc.c
)

set(VDB_SOURCES
  src/vdb/vdb.hh
  src/vdb/vdb_api.cc
  src/vdb/vdb_api.hh
  src/vdb/vdb_command_impl.cc
  src/vdb/data/checker.cc
  src/vdb/data/checker.hh
  src/vdb/data/checker_registry.cc
  src/vdb/data/checker_registry.hh
  src/vdb/data/statistics_collector.cc
  src/vdb/data/statistics_collector.hh
  src/vdb/common/defs.hh
  src/vdb/common/memory_allocator.cc
  src/vdb/common/memory_allocator.hh
  src/vdb/common/spinlock.cc
  src/vdb/common/spinlock.hh
  src/vdb/common/status.hh
  src/vdb/common/status.cc
  src/vdb/common/system_log.cc
  src/vdb/common/system_log.hh
  src/vdb/common/util.cc
  src/vdb/common/util.hh
  src/vdb/common/common_guards.hh
  src/vdb/compute/execution.cc
  src/vdb/compute/execution.hh
  src/vdb/data/embedding_store.cc
  src/vdb/data/embedding_store.hh
  src/vdb/data/expression.cc
  src/vdb/data/expression.hh
  src/vdb/data/index_handler.cc
  src/vdb/data/index_handler.hh
  src/vdb/data/index_info.cc
  src/vdb/data/index_info.hh
  src/vdb/data/label_info.hh
  src/vdb/data/metadata_checker.cc
  src/vdb/data/metadata.hh
  src/vdb/data/mutable_array.hh
  src/vdb/data/segmentation.hh
  src/vdb/data/segmentation.cc
  src/vdb/data/table.cc
  src/vdb/data/table.hh
  src/vdb/tests/util_for_test.hh
  src/vdb/tests/util_for_test.cc
  src/vdb/metrics/metric.hh
  src/vdb/metrics/metrics.hh
  src/vdb/metrics/metrics.cc
  src/vdb/metrics/metrics_api.hh
  src/vdb/metrics/metrics_api.cc
  src/vdb/metrics/metrics_collection.hh
  src/vdb/metrics/metrics_collection.cc
  src/vdb/metrics/metric_info.hh
  src/vdb/metrics/metric_info.cc
  src/vdb/metrics/performance_monitor.hh
  src/vdb/metrics/performance_monitor.cc
  src/vdb/metrics/timer.hh
  src/vdb/metrics/timer.cc
  src/vdb/tests/base_environment.hh
  src/vdb/common/fd_manager.hh
  src/vdb/common/fd_manager.cc
  src/vdb/common/json_validator.hh
  src/vdb/common/json_validator.cc
  src/vdb/common/server_configuration.hh
  src/vdb/common/server_configuration.cc
  src/vdb/data/primary_key.hh
  src/vdb/data/primary_key.cc
  src/vdb/data/primary_key_index.hh
  src/vdb/data/primary_key_index.cc
  src/vdb/index/hnsw/common/interface.hh
  src/vdb/index/hnsw/common/filter.hh
  src/vdb/index/hnsw/common/visited_list_pool.hh
  src/vdb/index/hnsw/hnsw.hh
  src/vdb/index/hnsw/bruteforce.hh
  src/vdb/vector/common/simd_header.hh
  src/vdb/vector/common/params.hh
  src/vdb/vector/distance/space_header.hh
  src/vdb/vector/distance/space_l2_float32.hh
  src/vdb/vector/distance/space_ip_float32.hh
)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add executable target for redis-server
add_executable(seahorse-server ${REDIS_SERVER_SOURCES} ${VDB_SOURCES})

# `add_dependencies()` enforces the order of compilations
add_dependencies(seahorse-server jemalloc_lib)

# add pthread into redis-server using 'target_compile_options'
target_compile_options(seahorse-server PRIVATE -pthread)

# for json library
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()

# Link necessary libraries for redis-server
# Note: pthread is handled by -pthread flag in target_compile_options
target_link_libraries(seahorse-server
    hiredis
    lua
    hdr_histogram
    linenoise
    fpconv
    ${IO_LIBRARIES}
    jemalloc_lib
    Arrow::arrow_shared
    ArrowDataset::arrow_dataset_shared
    ArrowCompute::arrow_compute_shared
    ArrowAcero::arrow_acero_shared
    nlohmann_json::nlohmann_json
    ${AWSSDK_LIBRARIES}
)

add_library(redis-lib ${REDIS_SERVER_SOURCES} ${VDB_SOURCES})
target_compile_definitions(redis-lib PRIVATE BUILDING_REDIS_LIB)
target_link_libraries(redis-lib
    hiredis
    lua
    hdr_histogram
    linenoise
    fpconv
    ${IO_LIBRARIES}
    jemalloc_lib
    Arrow::arrow_shared
    ArrowDataset::arrow_dataset_shared
    ArrowCompute::arrow_compute_shared
    ArrowAcero::arrow_acero_shared
    nlohmann_json::nlohmann_json
    ${AWSSDK_LIBRARIES}
)

# List of source files for seahorse-cli, derived from the object files mentioned
set(REDIS_CLI_SOURCES
    src/adlist.c
    src/ae.c
    src/anet.c
    src/cli_commands.c
    src/cli_common.c
    src/crc16.c
    src/crc64.c
    src/crcspeed.c
    src/dict.c
    src/monotonic.c
    src/mt19937-64.c
    src/redis-cli.c
    src/redisassert.c
    src/release.c
    src/release.h
    src/siphash.c
    src/strl.c
    src/zmalloc.c
)

# Add executable target for seahorse-cli
add_executable(seahorse-cli ${REDIS_CLI_SOURCES})
# Link necessary libraries for seahorse-cli
target_link_libraries(seahorse-cli hiredis linenoise m)

if (NOT WITH_TESTS STREQUAL "OFF")
  function(add_test test_name test_path)
    add_executable(${test_name} ${test_path})
    if (WITH_TESTS STREQUAL "DEBUG")
      target_compile_definitions(${test_name} PRIVATE _DEBUG_GTEST)
    endif()
    target_compile_definitions(${test_name} PRIVATE
        _REMOVE_ALL_AFTER_TEST=${_REMOVE_ALL_AFTER_TEST}
    )
    target_link_libraries(${test_name}
      GTest::gtest_main
      redis-lib
      hiredis
      lua
      hdr_histogram
      linenoise
      fpconv
      jemalloc_lib
      Arrow::arrow_shared
      ArrowDataset::arrow_dataset_shared
      ArrowCompute::arrow_compute_shared
      ArrowAcero::arrow_acero_shared
      nlohmann_json::nlohmann_json
    )
    set_target_properties(${test_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
    if (RUN_TESTS STREQUAL "ON")
      add_custom_command(
          TARGET ${test_name} POST_BUILD
          COMMENT "Running ${test_name}, ignore failures"
          COMMAND ${CMAKE_BINARY_DIR}/tests/${test_name} || true
      )
    endif()
    add_custom_command(
      TARGET ${test_name} PRE_BUILD
      COMMENT "Executing clang-format before building ${test_name}.cc"
      COMMAND clang-format -i --style=file `find ../src/vdb/tests/${test_name}.cc -type f`
    )
  endfunction()

  add_test("data_structure_tests" src/vdb/tests/data_structure_tests.cc)
  add_test("command_tests" src/vdb/tests/command_tests.cc)
  add_test("template_tests" src/vdb/tests/template_tests.cc)
  add_test("util_tests" src/vdb/tests/util_tests.cc)
  add_test("index_handler_tests" src/vdb/tests/index_handler_tests.cc)
  add_test("index_tests" src/vdb/tests/index_tests.cc)
  add_test("snapshot_tests" src/vdb/tests/snapshot_tests.cc)
  add_test("allocator_tests" src/vdb/tests/allocator_tests.cc)
  add_test("expression_tests" src/vdb/tests/expression_tests.cc)
  add_test("metric_tests" src/vdb/tests/metric_tests.cc)
  add_test("xxhash_tests" src/vdb/tests/xxhash_tests.cc)
  add_test("segment_tests" src/vdb/tests/segment_tests.cc)
  add_test("systemlog_tests" src/vdb/tests/systemlog_tests.cc)
  add_test("rollback_tests" src/vdb/tests/rollback_tests.cc)
endif()

if (APPLY_CLANG_FORMAT)
  add_custom_command(
      TARGET seahorse-server PRE_BUILD
      COMMENT "Executing clang-format before building"
      COMMAND clang-format -i --style=file `find ../src/vdb -type f`
  )
endif()
