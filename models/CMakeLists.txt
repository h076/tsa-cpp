# force CMake to generate compile commands
file(GLOB_RECURSE LINEARMODELS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(linearModels STATIC ${LINEARMODELS_SOURCES})

target_include_directories(linearModels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/linear
    ${PROJECT_SOURCE_DIR}/external/xtensor/include
    ${PROJECT_SOURCE_DIR}/external/xtensor-blas/include
)
