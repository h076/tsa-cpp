cmake_minimum_required(VERSION 3.19)
project(statsTesting LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# === Copy compile_commands.json to root ===
add_custom_target(copy_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_BINARY_DIR}/compile_commands.json
          ${PROJECT_SOURCE_DIR}/compile_commands.json
  COMMENT "Copying compile_commands.json to root directory"
)

# === Package required by xtensor-blas ===
find_package(LAPACK REQUIRED)

# === Add subdirectories first ===
add_subdirectory(linearModels)
add_subdirectory(tools)
add_subdirectory(tests)

# === Main executable ===
add_executable(${PROJECT_NAME} src/main.cpp)

# === Link libraries ===
target_link_libraries(${PROJECT_NAME} PRIVATE ${LAPACK_LIBRARIES} tests)
