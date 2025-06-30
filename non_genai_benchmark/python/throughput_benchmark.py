# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark statically shaped non-generative AI models with OpenVINO, with asynchronous inference optimized for throughput.

Modified from OpenVINO's througput_benchmark.py sample https://github.com/openvinotoolkit/openvino/tree/master/samples/python/benchmark/throughput_benchmark
This script adds:
- performance counters
- logging
- reshaping on the fly
- percentile metrics (in log)

Requirements: `pip install openvino`

- This is a sample script, not intended as a production-ready benchmarking application that works out of the box for every model. You are encouraged to read
  the code and modify for your purposes!
- The script only works for statically shaped models. For a dynamically shaped model, add for example `--reshape (1,3,512,512)` to reshape the model to static shapes before benchmarking.
- Use `--log logfile.csv` to log the results to an output file (one line per run). Results are APPENDED to the file if it already exists.
- To limit to one specific CPU core, use for example `--core 5`. This is experimental and requires `pip install psutil`.
- Logging performance counters has a small overhead. For small models, this can be significant. Measure actual performance without performance counters,
  and then measure with performance counters to find potential performance bottlenecks
- For performance counters only one request is measured
- It is often useful to run this in a loop with logging enabled and take the median of the median latencies
  - Windows: `for /L %%i in (1,1,5) do @python throughput_benchmark.py model.xml CPU --log log.csv`
  - Linux: `for i in {1..5}; do python throughput_benchmark.py model.xml CPU --log log.csv; done`

Usage: python thoughput_benchmark.py model device [--performance_counters] [--reshape (1,3,240,240)] [--config path_to_config.json] [--log log.csv]
"""
import argparse
import ast
import csv
import datetime
import gc
import importlib.metadata
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import openvino as ov


try:
    from openvino.utils.types import get_dtype
except ModuleNotFoundError:
    from openvino.runtime.utils.types import get_dtype


def fill_tensor_random(tensor):
    dtype = get_dtype(tensor.element_type)
    rand_min, rand_max = (0, 1) if dtype == bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ["i", "u", "b"]:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    if 0 == tensor.get_size():
        raise RuntimeError("Models with dynamic shapes aren't supported. Input tensors must have specific shapes before inference")
    tensor.data[:] = rs.uniform(rand_min, rand_max, list(tensor.shape)).astype(dtype)


def list_performance_counters(request, filename=None):
    """
    Show performance counters for the request. Model must have been compiled with PERF_COUNT enabled

    Params:
    - request: inference request
    - filename: Optional filename for .csv file to store all performance counters.
    """

    profiling_list = []
    for i in request.profiling_info:
        d = {
            "node_name": i.node_name,
            "node_type": i.node_type,
            "status": i.status,
            "exec_type": i.exec_type,
            "real_time": i.real_time.microseconds,
            "cpu_time": i.cpu_time.microseconds,
        }
        profiling_list.append(d)
    profiling_list.sort(key=lambda x: x["real_time"], reverse=True)
    print()
    print(" | ".join(f"{h:<20}" for h in d.keys()))

    for row in profiling_list[:5]:
        print(" | ".join(f"{str(row[h]):<20}"[:20] for h in d.keys()))

    if filename is not None:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d.keys())
            writer.writeheader()
            for row in profiling_list:
                writer.writerow(row)


def benchmark(model_or_file, device, user_config=None, reshape=None, performance_counters=False, logfile=None, limit_cpu_core=None):
    """
    Benchmark model_or_file on `device`, with asynchronous inference in THROUGHPUT mode.

    Params:
    - model: ov.Model or path to .onnx or .xml model
    - device: Inference device. CPU, GPU, NPU, AUTO
    - user_config: Optional path to json file with OpenVINO config which is passed to .compile_model
    - reshape: Optional shape to reshape the model to. Example `--reshape (1,3,512,512)`
    - performance_counters: set to True to show 5 most time consuming apps after benchmark
    - logfile: Optional path to logfile. Benchmark results will be appended to this file
    - limit_cpu_core: Optional integer number of the specific CPU core to limit the process to (requires `pip install psutil`)
    """
    if limit_cpu_core is not None:
        import psutil

        pid = os.getpid()
        p = psutil.Process(pid)
        p.cpu_affinity([limit_cpu_core])

    core = ov.Core()

    if isinstance(model_or_file, str) or isinstance(model_or_file, os.PathLike):
        model = core.read_model(model_or_file)
        modelfile = model_or_file
    else:
        model = model_or_file
        try:
            modelfile = model.get_rt_info("model_name").value
        except RuntimeError:
            modelfile = model.get_friendly_name()

    if model.inputs[0].get_partial_shape().is_dynamic and reshape is None:
        raise ValueError("Models with dynamic shapes are not supported. To reshape to static shapes on the fly, use the `reshape` argument")
    if reshape is not None:
        model.reshape(reshape)

    performance_hint = "THROUGHPUT" if not "AUTO" in device else "CUMULATIVE_THROUGHPUT"
    config = {"PERFORMANCE_HINT": performance_hint}
    print(config)
    if performance_counters:
        config["PERF_COUNT"] = performance_counters

    if user_config is not None:
        with open(user_config) as f:
            loaded_config = json.load(f)
            config.update(loaded_config)

    # TODO this is not robust and doesn't work correctly with GPU.1 etc, but this is only for logging
    cpu_device = core.get_property("CPU", "FULL_DEVICE_NAME")
    device_info = cpu_device
    if "GPU" in device:
        device_info += ", " + core.get_property("GPU", "FULL_DEVICE_NAME")
    if "NPU" in device:
        device_info += ", " + core.get_property("NPU", "FULL_DEVICE_NAME")

    compiled_model = core.compile_model(model, device, config)

    ireqs = ov.AsyncInferQueue(compiled_model)
    # Fill input data for ireqs
    for ireq in ireqs:
        for model_input in compiled_model.inputs:
            fill_tensor_random(ireq.get_tensor(model_input))
    # Warm up
    for _ in range(len(ireqs)):
        ireqs.start_async()
    ireqs.wait_all()

    # Benchmark for seconds_to_run seconds and at least niter iterations
    seconds_to_run = 20
    niter = 10
    latencies = []
    in_fly = set()
    start = perf_counter()
    time_point_to_finish = start + seconds_to_run
    while perf_counter() < time_point_to_finish or len(latencies) + len(in_fly) < niter:
        idle_id = ireqs.get_idle_request_id()
        if idle_id in in_fly:
            latencies.append(ireqs[idle_id].latency)
        else:
            in_fly.add(idle_id)
        ireqs.start_async()
    ireqs.wait_all()
    duration = perf_counter() - start
    for infer_request_id in in_fly:
        latencies.append(ireqs[infer_request_id].latency)

    throughput = len(latencies) / duration

    # Print and optionally save log
    version = importlib.metadata.version("openvino")
    d = {
        "datetime": datetime.datetime.now().isoformat(),
        "model": modelfile,
        "device": device,
        "device_name": device_info,
        "config": config,
        "count": len(latencies),
        "duration": round(duration * 1000, 2),
        "latency_median": round(np.median(latencies), 2),
        "latency_p90": round(np.percentile(latencies, 90), 2),
        "latency_p95": round(np.percentile(latencies, 95), 2),
        "latency_p99": round(np.percentile(latencies, 99), 2),
        "latency_mean": round(sum(latencies) / len(latencies), 2),
        "latency_max": round(max(latencies), 2),
        "throughput": round(throughput, 2),
        "openvino": version,
    }
    if logfile is not None:
        writeheader = not Path(logfile).is_file()

        with open(logfile, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d.keys())
            if writeheader:
                writer.writeheader()
            writer.writerow(d)

    # TODO change device logging for use with dGPU
    configlog = f"{loaded_config}  " if user_config is not None else ""
    print(f'{"model":<27}{"device":<16}{"system":<35}{"openvino":<22}{"throughput":<16}{"config" if configlog else ""}')
    print(f'{d["model"]:<27}{d["device"]:<16}{cpu_device:<35}{d["openvino"]:<22}{d["throughput"]:<16}{configlog}')
    if performance_counters:
        list_performance_counters(ireqs[0], filename=None if logfile is None else filename)

    del compiled_model
    gc.collect()


def benchmark_all(model_or_file, performance_counters, reshape, user_config, logfile, limit_cpu_core):
    devices = ov.Core().available_devices
    if len(devices) > 1:
        devices.append("AUTO")
 
    if "NPU" in devices:
        # Move NPU to the end of the list so that if it crashes only NPU inference is not run
        devices = [device for device in devices if "NPU" not in device] + ["NPU"]
    for device in devices:
        benchmark(
            model_or_file=model_or_file,
            device=device,
            user_config=user_config,
            reshape=reshape,
            performance_counters=performance_counters,
            logfile=logfile,
            limit_cpu_core=limit_cpu_core,
        )


def parse_shape(shape):
    try:
        value = ast.literal_eval(shape)
    except ValueError:
        value = None
    if not (isinstance(value, (list, tuple)) and all(isinstance(i, int) for i in value)):
        raise argparse.ArgumentTypeError("Shape must be a list or tuple of integers")
    return tuple(value)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("device")
    parser.add_argument("--performance_counters", "-pc", action="store_true")
    parser.add_argument("--reshape")
    parser.add_argument("--config")
    parser.add_argument("--log")
    parser.add_argument("--core", type=int)

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    reshape = parse_shape(args.reshape) if args.reshape is not None else None
    print(args)

    if args.device.lower() == "all":
        benchmark_all(
            model_or_file=args.model,
            user_config=args.config,
            reshape=reshape,
            performance_counters=args.performance_counters,
            logfile=args.log,
            limit_cpu_core=args.core,
        )
    else:
        benchmark(
            model_or_file=args.model,
            device=args.device,
            user_config=args.config,
            reshape=reshape,
            performance_counters=args.performance_counters,
            logfile=args.log,
            limit_cpu_core=args.core,
        )
