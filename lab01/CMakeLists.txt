cmake_minimum_required(VERSION 3.20)

project(lab01 CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lab01
    src/main.cpp
)

target_include_directories(lab01 PRIVATE
    include
)