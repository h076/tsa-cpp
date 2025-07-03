file(GLOB_RECURSE TESTS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(tests STATIC ${TESTS_SOURCES})

target_include_directories(tests PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/external/xtensor/include
)

target_link_libraries(tests PUBLIC tools linearModels)
