cmake_minimum_required(VERSION 3.16)
project(robust_spline_fitter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

execute_process(COMMAND python3 -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(rsf src/bindings.cpp src/cardinal_spline.cpp src/spline_distance_analytic.cpp
                     src/spline_distance_sampled.cpp src/ransac_fitter.cpp)
target_include_directories(rsf PRIVATE include)