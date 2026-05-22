cmake_minimum_required(VERSION 3.10)
project(encrypted_model_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")

find_package(OpenVINO REQUIRED)
find_package(OpenSSL REQUIRED)

add_executable(model_crypto main.cpp)
target_link_libraries(model_crypto PRIVATE openvino::runtime OpenSSL::Crypto)
