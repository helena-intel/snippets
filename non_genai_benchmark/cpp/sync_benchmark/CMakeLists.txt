# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Standalone build: cmake -B build && cmake --build build
# Requires that OpenVINO environment is set up (setupvars.bat/sh has been run)

cmake_minimum_required(VERSION 3.16)
project(sync_benchmark)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_executable(sync_benchmark main.cpp)

target_link_libraries(sync_benchmark PRIVATE openvino::runtime)

