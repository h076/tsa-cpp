file(GLOB_RECURSE TOOLS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(tools STATIC ${TOOLS_SOURCES})

target_include_directories(tools PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/external/xtensor/include
)

target_link_libraries(tools PUBLIC linearModels)

find_package(LAPACK REQUIRED)

add_executable(coreToolsTest src/coreToolsTests.cpp)
target_link_libraries(coreToolsTest tools ${LAPACK_LIBRARIES})
