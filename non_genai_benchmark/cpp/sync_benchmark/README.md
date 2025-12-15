# sync_benchmark (standalone)

Small synchronous inference benchmark based on OpenVINO's sync_benchmark sample.
This repository contains a standalone copy of the sample with a small addition: a `--cache-dir` argument
that sets the OpenVINO runtime cache directory.

## Prerequisites

- OpenVINO is installed and environment variables are set (run `setupvars.bat` on Windows or `setupvars.sh` on Linux/macOS).
- CMake 3.16 or newer and a C++ compiler supporting C++11.


### Example (Windows)

In a Visual Studio Developer Command Prompt, run the following lines to download OpenVINO and set the environment variables:

```
curl -O https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4/windows/openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64.zip
tar -xf openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64.zip
move openvino_toolkit_windows_2025.4.0.20398.8fdad55727d_x86_64 openvino_20254
openvino_20254\setupvars.bat
```

Run the following lines to download the sample without cloning the repository:

```
curl -O https://raw.githubusercontent.com/helena-intel/snippets/refs/heads/main/non_genai_benchmark/cpp/sync_benchmark/CMakeLists.txt
curl -O https://raw.githubusercontent.com/helena-intel/snippets/refs/heads/main/non_genai_benchmark/cpp/sync_benchmark/main.cpp
```

And then go to the next step to build the sample.

## Build

From this directory (where `CMakeLists.txt` and `main.cpp` live) run:

```bash
cmake -B build && cmake --build build --config Release
```

This produces the `sync_benchmark` executable in the `build` output tree.

## Usage

```
sync_benchmark <path_to_model> [device_name] [--cache-dir <path>]
```

- `path_to_model`: path to an OpenVINO IR or supported model file (for example, an `.xml`/.bin pair or an ONNX file depending on your runtime).
- `device_name`: optional device name (default: `CPU`, options: `CPU`, `GPU`, `NPU`)
- `--cache-dir <path>`: optional. If provided, sets the OpenVINO runtime `cache_dir` configuration option to the specified path.

Example:

```bash
# Run on CPU with a cache directory
sync_benchmark model.xml CPU --cache-dir C:/tmp/ov_cache
```

> **NOTE:** When using `--cache-dir`, the first time you load a model the runtime will create and populate the cache
> which can use additional disk I/O and memory and will take longer. Subsequent runs that use the same cache directory will
> load the model faster from the cache.

## What this sample does
- Loads and compiles the provided model on the selected device.
- Fills model inputs with random data.
- Runs a warm-up inference and then measures synchronous inference latency for a default duration (10 seconds) and at least 10 iterations.
- Reports count, duration, average and median latency, and throughput (FPS).

## Origin and License
This sample is based on the OpenVINO `sync_benchmark` sample:
https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp/benchmark/sync_benchmark

SPDX-License-Identifier: Apache-2.0
