file(GLOB_RECURSE TESTS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(tests STATIC ${TESTS_SOURCES})

target_include_directories(tests PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/external/xtensor/include
)

target_link_libraries(tests PUBLIC tools linearModels)

find_package(LAPACK REQUIRED)

add_executable(testADFT src/testADFT.cpp)
target_link_libraries(testADFT PRIVATE tests ${LAPACK_LIBRARIES})

add_executable(testHurst src/testHurst.cpp)
target_link_libraries(testHurst PRIVATE tests ${LAPACK_LIBRARIES})
