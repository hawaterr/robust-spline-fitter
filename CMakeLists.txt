cmake_minimum_required(VERSION 3.16)
project(robust_spline_fitter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(fit_spline src/main.cpp src/cardinal_spline.cpp src/ransac_fitter.cpp) # Learn: add_executable tells CMake which .cpp files to compile into object files and link together into the fit_spline binary.
target_include_directories(fit_spline PRIVATE include) # Learn: A target is a buildable unit in CMake — an executable or a library