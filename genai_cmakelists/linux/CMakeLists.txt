cmake_minimum_required(VERSION 3.23)
project(OpenVINOChatbotExample)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenVINOGenAI REQUIRED)
add_executable(openvinochat openvinochat.cpp)
target_link_libraries(openvinochat PRIVATE openvino::genai)

set_target_properties(openvinochat PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(openvinochat PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

set(INTEL_OPENVINO_DIR $ENV{INTEL_OPENVINO_DIR})
set(OPENVINO_LIB_DIR "${INTEL_OPENVINO_DIR}/runtime/lib/intel64")
set(TBB_LIB_DIR "${INTEL_OPENVINO_DIR}/runtime/3rdparty/tbb/lib")

install(TARGETS openvinochat RUNTIME DESTINATION .)
install(FILES
      "${OPENVINO_LIB_DIR}/libopenvino.so"
      "${OPENVINO_LIB_DIR}/libopenvino.so.2024.6.0"
      "${OPENVINO_LIB_DIR}/libopenvino.so.2460"
      "${OPENVINO_LIB_DIR}/libopenvino_ir_frontend.so.2024.6.0"
      "${OPENVINO_LIB_DIR}/libopenvino_ir_frontend.so.2460"
      "${OPENVINO_LIB_DIR}/libopenvino_intel_cpu_plugin.so"
      "${OPENVINO_LIB_DIR}/libopenvino_genai.so.2024.6.0.0"
      "${OPENVINO_LIB_DIR}/libopenvino_genai.so.2460"
      "${OPENVINO_LIB_DIR}/libopenvino_genai.so"
      "${OPENVINO_LIB_DIR}/libopenvino_tokenizers.so"
      "${OPENVINO_LIB_DIR}/libcore_tokenizers.so"
      "${TBB_LIB_DIR}/libtbb.so.12"
      "${TBB_LIB_DIR}/libtbb.so.12.13"
      "${TBB_LIB_DIR}/libtbb.so"
      "${TBB_LIB_DIR}/libtbbmalloc.so"
      "${TBB_LIB_DIR}/libtbbmalloc.so.2"
      "${TBB_LIB_DIR}/libtbbmalloc.so.2.13"
      "${TBB_LIB_DIR}/libtbbmalloc_proxy.so"
      "${TBB_LIB_DIR}/libtbbmalloc_proxy.so.2"
      "${TBB_LIB_DIR}/libtbbmalloc_proxy.so.2.13"
      "${TBB_LIB_DIR}/libtbbbind_2_5.so"
      "${TBB_LIB_DIR}/libtbbbind_2_5.so.3"
      "${TBB_LIB_DIR}/libtbbbind_2_5.so.3.13"
      "${TBB_LIB_DIR}/libhwloc.so.15.6.4"
      "${TBB_LIB_DIR}/libhwloc.so.15"
      "${TBB_LIB_DIR}/libhwloc.so"
      "${OPENVINO_LIB_DIR}/cache.json"
      "${OPENVINO_LIB_DIR}/libopenvino_intel_gpu_plugin.so"
DESTINATION .)
