cmake_minimum_required(VERSION 3.27)
project(micrograd)

set(CMAKE_CXX_STANDARD 23)

add_executable(micrograd++ main.cpp
        engine.h
        engine.cpp
        visualization.h
        visualization.cpp
        neuronet.h
        neuronet.cpp)
