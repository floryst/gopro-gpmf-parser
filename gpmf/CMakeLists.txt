cmake_minimum_required(VERSION 3.12)

project(gpmf)

add_executable(${PROJECT_NAME} gpmf/main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Werror)