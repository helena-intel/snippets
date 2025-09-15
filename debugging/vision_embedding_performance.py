"""
Show performance counters of vision embeddings model (part of VLM).
Tested with InternVL, SmolVLM, LLaVA, MiniCPM-V-2_6, gemma-3. Script does not work with Qwen and Phi models.

Usage: python __file__ /path/to/model_dir DEVICE [logfile.csv]
model_dir should be a directory containing OpenVINO VLM models (.xml and .bin)
logfile is optional, if passed the performance counters of the first inference will be saved to that file.
by default the five most time consuming operations are printed to the screen.

Python requirements: `pip install openvino`
Model example: https://huggingface.co/echarlaix/SmolVLM-256M-Instruct-openvino
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import openvino as ov


def list_performance_counters(request, logfile=None):
    """
    Show performance counters for the request. Model must have been compiled with PERF_COUNT enabled

    Params:
    - request: inference request
    - logfile: Optional file name for .csv file to store all performance counters. Will be overwritten if exists.
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

    # Compute percentage of real_time for operation. An indication, not exact.
    total_real_time = sum([item["real_time"] for item in profiling_list])
    for item in profiling_list:
        item["percentage_time"] = round((item["real_time"] / total_real_time) * 100, 2)

    profiling_list.sort(key=lambda x: x["real_time"], reverse=True)
    profiling_list_print = [{k: v for k, v in d.items() if k != "status"} for d in profiling_list]

    # determine column widths for printing output (without requiring extra dependencies)
    max_width = 130
    col_keys = profiling_list_print[0].keys()
    widths = {}
    for k in col_keys:
        max_value_len = max((len(str(row.get(k, ""))) for row in profiling_list_print[:5]), default=0)
        widths[k] = min(max(len(k), max_value_len), max_width)

    print(" | ".join(f"{k:<{widths[k]}}" for k in col_keys))
    for row in profiling_list_print[:5]:
        print(" | ".join(f"{str(row.get(k, '')):<{widths[k]}}" for k in col_keys))

    if logfile is not None:
        with open(logfile, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d.keys())
            writer.writeheader()
            for row in profiling_list:
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model directory containing openvino_vision_embeddings_model.xml")
    parser.add_argument("device", help='Device to run on (e.g., "CPU", "GPU")')
    parser.add_argument(
        "--log", "-l", help="Path to an optional CSV file to log performance counters. Will be overwritten if exists"
    )
    args = parser.parse_args()

    num_runs = 1
    model_path = Path(args.model_path)
    device = args.device.upper()
    logfile = args.log

    model_xml = "openvino_vision_embeddings_model.xml"

    with open(model_path / "config.json") as f:
        config = json.load(f)
    image_size = config["vision_config"]["image_size"]
    patch_size = config["vision_config"]["patch_size"]
    mask_size = image_size // patch_size  # not robustly tested
    position_size = mask_size * mask_size
    num_images = 1

    if "minicpm" in str(model_path).lower():
        # hardcode minicpm for now. tested with openbmb/MiniCPM-V-2_6
        pixel_values_shape = (num_images, 3, 14, 14336)
        attention_mask_shape = (num_images, 1, 1024)
        patch_position_ids_shape = (num_images, 1024)
    else:
        pixel_values_shape = (num_images, 3, image_size, image_size)
        # attention_mask and patch_position_ids shapes are tested for smolvlm
        attention_mask_shape = (num_images, mask_size, mask_size)
        patch_position_ids_shape = (num_images, position_size)

    model = ov.compile_model(model_path / model_xml, device_name=device, config={"PERF_COUNT": "YES"})

    pixel_values = np.random.rand(*pixel_values_shape).astype(np.float32)
    patch_attention_mask = np.ones(attention_mask_shape, dtype=np.bool)
    patch_position_ids = np.random.randint(low=0, high=position_size - 1, size=patch_position_ids_shape, dtype=np.int64)
    model_inputs = [
        pixel_values,
    ]
    input_names = [i.any_name for i in model.inputs]
    if "patch_attention_mask" in input_names:
        model_inputs.append(patch_attention_mask)
    if "patch_position_ids" in input_names or "position_ids" in input_names:
        model_inputs.append(patch_position_ids)

    print("Input shape:", [item.shape for item in model_inputs])

    for i in range(1, num_runs + 1):
        ireq = model.create_infer_request()
        start = time.perf_counter()
        ireq.infer(model_inputs)
        end = time.perf_counter()
        print(f"Inference {i} duration ({device}): {end-start:.2f} seconds")
        print()
        list_performance_counters(ireq, logfile=logfile if i == 1 else None)
