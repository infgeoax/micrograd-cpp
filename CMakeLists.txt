cmake_minimum_required(VERSION 3.27)
project(micrograd)

set(CMAKE_CXX_STANDARD 23)

add_subdirectory(matplotplusplus)

add_library(micrograd engine.h engine.cpp visualization.h visualization.cpp neuronet.h neuronet.cpp)

add_executable(micrograd++ main.cpp)
target_link_libraries(micrograd++ micrograd)

add_executable(demo demo.cpp)
target_link_libraries(demo micrograd matplot)
