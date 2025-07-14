# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Example script that extends sync_benchmark.py to use HETERO plugin for faster inference.
This implementation is specifically useful for detection models with a TopK operation.

To use HETERO, on systems with GPU or NPU, use device HETERO:GPU,CPU or HETERO:NPU,CPU
The TopK operation will be run on the last device in the list. In the above examples, it will run on CPU.

Requires: sync_benchmark.py script and `pip install openvino`

Usage: python sync_benchmark_hetero.py model device [--performance_counters] [--reshape (1,3,240,240)] [--config path_to_config.json] [--log log.csv]
"""

import openvino as ov


def set_affinity(model, node_name, inference_device, target_device, model_name=None):
    """
    Modify `model` in place to set affinity to run `node_name` on `target_device`
    """
    supported_ops = ov.Core().query_model(model, inference_device)
    for node in model.get_ops():
        affinity = supported_ops[node.get_friendly_name()]
        if node_name in node.get_friendly_name():
            node.get_rt_info()["affinity"] = target_device
            for output in node.outputs():
                output.get_rt_info()["affinity"] = target_device
            for input in node.inputs():
                input.get_rt_info()["affinity"] = target_device
        else:
            node.get_rt_info()["affinity"] = affinity
            for output in node.outputs():
                output.get_rt_info()["affinity"] = affinity
            for input in node.inputs():
                input.get_rt_info()["affinity"] = affinity
    if model_name is not None:
        # set model_name to model's rt_info so we can retrieve it later
        model.set_rt_info(model_name, "model_name")


def benchmark_all(model_or_file, performance_counters, reshape, user_config, logfile, limit_cpu_core):
    devices = ov.Core().available_devices
    no_cpu_devices = [device for device in devices if device != "CPU"]
    for device in no_cpu_devices:
        devices.append(f"HETERO:{device},CPU")
    if "NPU" in devices:
        # Move NPU to the end of the list so that if it crashes only NPU inference is not run
        devices = [device for device in devices if "NPU" not in device] + ["HETERO:NPU,CPU", "NPU"]
    for device in devices:
        benchmark(
            model_or_file=model_or_file,
            device=device,
            reshape=reshape,
            user_config=user_config,
            performance_counters=performance_counters,
            logfile=logfile,
            limit_cpu_core=limit_cpu_core,
        )


if __name__ == "__main__":
    from sync_benchmark import benchmark, build_argparser, parse_shape

    parser = build_argparser()
    args = parser.parse_args()
    reshape = parse_shape(args.reshape) if args.reshape is not None else None

    model = ov.Core().read_model(args.model)

    # reshape before changing affinity
    if reshape is not None:
        model.reshape(reshape)

    hetero_devices = args.device.split(":")[-1].split(",")
    if args.device != "ALL" and not ("HETERO" in args.device and len(hetero_devices) > 1):
        raise ValueError("This sample should be used with HETERO device, specifying devices to run on. Example: 'HETERO:GPU,CPU'")


    if args.device.lower() == "all":
        benchmark_all(
            model_or_file=model,
            user_config=args.config,
            reshape=None,
            performance_counters=args.performance_counters,
            logfile=args.log,
            limit_cpu_core=args.core,
        )
    else:
        target_device = hetero_devices[-1]
        set_affinity(model=model, node_name="TopK", inference_device=args.device, target_device=target_device, model_name=args.model)
        benchmark(
            model_or_file=model,
            device=args.device,
            user_config=args.config,
            reshape=None,
            performance_counters=args.performance_counters,
            logfile=args.log,
            limit_cpu_core=args.core,
        )
