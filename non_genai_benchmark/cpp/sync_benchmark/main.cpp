// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: This file is a sample benchmark for demonstration purposes only.
// It is not intended for production use. The code prioritizes clarity
// and simplicity over robustness, security, or performance hardening.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "openvino/openvino.hpp"

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

void fill_tensor_random(ov::Tensor& tensor) {
    std::mt19937 gen(0);
    ov::element::Type type = tensor.get_element_type();
    size_t size = tensor.get_size();

    if (type == ov::element::f32) {
        auto* data = tensor.data<float>();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < size; i++) data[i] = dist(gen);
    } else if (type == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < size; i++) data[i] = ov::float16(dist(gen));
    } else if (type == ov::element::i64) {
        auto* data = tensor.data<int64_t>();
        std::uniform_int_distribution<int64_t> dist(0, 100);
        for (size_t i = 0; i < size; i++) data[i] = dist(gen);
    } else if (type == ov::element::i32) {
        auto* data = tensor.data<int32_t>();
        std::uniform_int_distribution<int32_t> dist(0, 100);
        for (size_t i = 0; i < size; i++) data[i] = dist(gen);
    } else if (type == ov::element::i8) {
        auto* data = tensor.data<int8_t>();
        std::uniform_int_distribution<int32_t> dist(-128, 127);
        for (size_t i = 0; i < size; i++) data[i] = static_cast<int8_t>(dist(gen));
    } else if (type == ov::element::u8) {
        auto* data = tensor.data<uint8_t>();
        std::uniform_int_distribution<int32_t> dist(0, 255);
        for (size_t i = 0; i < size; i++) data[i] = static_cast<uint8_t>(dist(gen));
    } else {
        // Fallback: fill with zeros
        memset(tensor.data(), 0, tensor.get_byte_size());
    }
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "OpenVINO:\n" << ov::get_openvino_version() << std::endl;

        std::string device_name = "CPU";
        std::string cache_dir;

        // Parse arguments
        for (int i = 2; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--cache-dir" && i + 1 < argc) {
                cache_dir = argv[++i];
            } else if (device_name == "CPU") {
                device_name = arg;
            }
        }

        if (argc < 2) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> [device_name] [--cache-dir <path>]" << std::endl;
            return EXIT_FAILURE;
        }

        ov::AnyMap ov_config{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};
        if (!cache_dir.empty()) {
            ov_config[ov::cache_dir.name()] = cache_dir;
        }

        // for testing: set other OpenVINO runtime options manually
        // ov_config[ov::inference_num_threads.name()] = 1;
        // ov_config[ov::compilation_num_threads.name()] = 1;

        // Print configured OpenVINO runtime options
        std::cout << "OpenVINO config:" << std::endl;
        for (const auto& kv : ov_config) {
            std::cout << "  " << kv.first << " : " << kv.second.as<std::string>() << std::endl;
        }

        // Create ov::Core and use it to compile a model.
        // Select the device by providing the name as the second parameter to CLI.
        ov::Core core;
        ov::CompiledModel compiled_model = core.compile_model(argv[1], device_name, ov_config);

        // Reject models with dynamic input shapes. Currently only statically shaped models are supported.
        for (const auto& model_input : compiled_model.inputs()) {
            const ov::PartialShape& pshape = model_input.get_partial_shape();
            if (!pshape.is_static()) {
                std::cerr << "Error: dynamic input shapes are not supported by this benchmark."
                          << " Please use a model with static input shapes." << std::endl;
                return EXIT_FAILURE;
            }
        }

        ov::InferRequest ireq = compiled_model.create_infer_request();
        // Fill input data for the ireq
        for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
            ov::Tensor tensor = ireq.get_tensor(model_input);
            fill_tensor_random(tensor);
        }
        // Warm up
        ireq.infer();
        // Benchmark for seconds_to_run seconds and at least niter iterations
        std::chrono::seconds seconds_to_run{20};
        size_t niter = 10;
        std::vector<double> latencies;
        latencies.reserve(niter);
        auto start = std::chrono::steady_clock::now();
        auto time_point = start;
        auto time_point_to_finish = start + seconds_to_run;
        while (time_point < time_point_to_finish || latencies.size() < niter) {
            ireq.infer();
            auto iter_end = std::chrono::steady_clock::now();
            latencies.push_back(std::chrono::duration_cast<Ms>(iter_end - time_point).count());
            time_point = iter_end;
        }
        auto end = time_point;
        double duration = std::chrono::duration_cast<Ms>(end - start).count();
        // Calculate latency statistics
        std::sort(latencies.begin(), latencies.end());
        double median = latencies[latencies.size() / 2];
        double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        // Report results
        std::cout << "Count:      " << latencies.size() << " iterations" << std::endl;
        std::cout << "Duration:   " << std::fixed << std::setprecision(2) << duration << " ms" << std::endl;
        std::cout << "Latency:    " << std::fixed << std::setprecision(2)
                  << "AVG=" << avg << " ms, MED=" << median << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                  << (latencies.size() * 1000.0 / duration) << " FPS" << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
