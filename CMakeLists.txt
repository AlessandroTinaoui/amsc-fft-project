cmake_minimum_required(VERSION 3.20)

project(fourier_pricer LANGUAGES CXX)

add_library(fourier_pricer src/fourier/DirectTransform.cpp)
target_compile_features(fourier_pricer PUBLIC cxx_std_20)
target_include_directories(fourier_pricer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(CTest)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG 03597a01ee50ed33e9dfd640b249b4be3799d395)
    FetchContent_MakeAvailable(googletest)
    add_executable(direct_transform_tests tests/DirectTransformTests.cpp)
    target_link_libraries(direct_transform_tests PRIVATE fourier_pricer GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(direct_transform_tests)
endif()
