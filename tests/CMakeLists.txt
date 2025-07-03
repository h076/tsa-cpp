add_executable(testLagmat src/testLagmat.cpp)

target_link_libraries(testLagmat PRIVATE tests)

add_executable(testAddTrend src/testAddTrend.cpp)

target_link_libraries(testAddTrend PRIVATE tests)
