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

# tools header library
add_library(tools INTERFACE)
target_include_directories(tools INTERFACE
    ${PROJECT_SOURCE_DIR}/tools/src
    ${PROJECT_SOURCE_DIR}/external/xtensor/include
)

# linear models library
add_library(linearModels INTERFACE)
target_include_directories(linearModels INTERFACE
    ${PROJECT_SOURCE_DIR}/models/src
    ${PROJECT_SOURCE_DIR}/external/xtensor/include
)

target_link_libraries(tests INTERFACE tools linearModels)

add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(linearModels)

# Main executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE tests)
