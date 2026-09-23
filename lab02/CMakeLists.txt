cmake_minimum_required(VERSION 4.1)
project(lab02 CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(matrix_mul src/main.cpp src/matrix.cpp)
target_include_directories(matrix_mul PRIVATE include)

find_package(OpenMP REQUIRED)
target_link_libraries(matrix_mul PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
  target_compile_options(matrix_mul PRIVATE /W4)
else()
  target_compile_options(matrix_mul PRIVATE -Wall -Wextra -Wpedantic)
endif()