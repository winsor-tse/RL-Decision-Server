cmake_minimum_required(VERSION 3.16)
project(YugenSimulation LANGUAGES CXX)

add_library(yugen_sim INTERFACE)
target_include_directories(yugen_sim INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(yugen_sim INTERFACE cxx_std_17)

add_executable(yugen_sim_example main.cpp)
target_link_libraries(yugen_sim_example PRIVATE yugen_sim)
