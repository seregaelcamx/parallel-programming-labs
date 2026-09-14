cmake_minimum_required(VERSION 4.1)
project(lab01_matrix_mul LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(matrix_mul
  src/main.cpp
  src/matrix.cpp
)
target_include_directories(matrix_mul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(matrix_mul PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)