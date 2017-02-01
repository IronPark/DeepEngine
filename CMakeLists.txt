cmake_minimum_required(VERSION 3.6)
project(DeepEngine)

set(CMAKE_C_STANDARD 99)

set(SOURCE_FILES main.c neuron.c neuron.h)
add_executable(DeepEngine ${SOURCE_FILES})