cmake_minimum_required(VERSION 3.19)
project(statsTesting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# test header library
add_library(tests INTERFACE)
target_include_directories(tests INTERFACE
    ${PROJECT_SOURCE_DIR}/tests/src
    ${PROJECT_SOURCE_DIR}/external/xtensor/include  # xtensor manually added
)

# Main executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tests)
