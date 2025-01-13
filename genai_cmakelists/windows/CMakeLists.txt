# CMakeLists.txt for building and installing OpenVINO GenAI projects
# This basic example copies all required OpenVINO libraries
# to the installation directory, together with the executable.
# Change SOURCE_FILE to your C++ source file name.
#
# The target system needs to have MS Visual C++ Redistributable
# https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
# GPU drivers need to be installed for GPU support; for NPU support, NPU drivers
# need to be installed, and the openvino_intel_npu_plugin.dll line must be uncommented

cmake_minimum_required(VERSION 3.23)
project(OpenVINOExample)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenVINOGenAI REQUIRED)

set(SOURCE_FILE openvinochat.cpp)
get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
add_executable(${TARGET_NAME} ${SOURCE_FILE})
target_link_libraries(${TARGET_NAME} PRIVATE openvino::genai)

string(REPLACE "\\" "/" INTEL_OPENVINO_DIR $ENV{INTEL_OPENVINO_DIR})
set(OPENVINO_LIB_DIR "${INTEL_OPENVINO_DIR}/runtime/bin/intel64/Release")
set(TBB_LIB_DIR "${INTEL_OPENVINO_DIR}/runtime/3rdparty/tbb/bin")

install(TARGETS ${TARGET_NAME} RUNTIME DESTINATION .)
install(FILES
      "${OPENVINO_LIB_DIR}/openvino.dll"
      "${OPENVINO_LIB_DIR}/openvino_ir_frontend.dll"
      "${OPENVINO_LIB_DIR}/openvino_intel_cpu_plugin.dll"
      "${OPENVINO_LIB_DIR}/openvino_genai.dll"
      "${OPENVINO_LIB_DIR}/openvino_tokenizers.dll"
      "${OPENVINO_LIB_DIR}/core_tokenizers.dll"
      "${OPENVINO_LIB_DIR}/icudt70.dll"
      "${OPENVINO_LIB_DIR}/icuuc70.dll"
      "${TBB_LIB_DIR}/tbb12.dll"
      "${TBB_LIB_DIR}/tbbmalloc.dll"
      "${TBB_LIB_DIR}/tbbmalloc_proxy.dll"
      "${TBB_LIB_DIR}/tbbbind_2_5.dll"
      "${OPENVINO_LIB_DIR}/cache.json"
      "${OPENVINO_LIB_DIR}/openvino_intel_gpu_plugin.dll"
      # "${OPENVINO_LIB_DIR}/openvino_intel_npu_plugin.dll"
DESTINATION .)
