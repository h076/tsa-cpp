cmake_minimum_required(VERSION 3.19)
project(tsa-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Copy compile_commands.json to root ===
if(NOT TARGET copy_compile_commands)
    add_custom_target(copy_compile_commands ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_SOURCE_DIR}/compile_commands.json
        COMMENT "Copying compile_commands.json to root directory"
    )
endif()

# === Find xtensor and blas ===
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(LAPACK)

set(TSA_DEPS
  xtensor
  xtensor-blas
  lapack
)

# === Set up library ===
add_library(tsa INTERFACE)


# === Expose include as interface ===
target_include_directories(tsa
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(tsa INTERFACE ${TSA_DEPS})
