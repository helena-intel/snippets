#!/usr/bin/env python
"""
Example script to show model conversion properties of OpenVINO LLM models converted with optimum-intel.

Usage: python modelinfo.py /path/to/model_directory_or_xml_file
Requires OpenVINO: `pip install openvino`

If a model_directory is given, the model file is assumed to be openvino_model.xml (this is the default for LLMs). 
For other models, pass the full path to the model's xml file.

More info: https://medium.com/openvino-toolkit/quick-checks-big-insights-debugging-openvino-models-with-rt-info-52ea86e35e95
"""

import sys
from pathlib import Path

import openvino as ov

if len(sys.argv) != 2:
    raise ValueError(f"Usage: {sys.argv[0]} /path/to/model_directory_or_xml_file")

model_path = Path(sys.argv[1])
if model_path.is_file() and model_path.suffix == ".xml":
    ov_model_path = model_path
elif model_path.is_dir() and (model_path / "openvino_model.xml").is_file():
    ov_model_path = model_path / "openvino_model.xml"
else:
    raise ValueError(
        f"model_path should point to an OpenVINO .xml file, or a model directory containing openvino_model.xml"
    )


def print_rt_info(model: ov.Model, key: str | list) -> None:
    """
    Prints rt_info for a model for the given key. `key` can be a single key, e.g. "optimum"
    or a list, e.g. ["nncf", "weight_compression"].
    """
    if model.has_rt_info(key):
        rt_value = model.get_rt_info(key).value
        key = key[-1] if isinstance(key, list) else key
        if isinstance(rt_value, dict):
            for key, value in rt_value.items():
                print(f"{key:<25}: {value}")
            print()
        else:
            print(f"{key:<25}: {rt_value}")


model = ov.Core().read_model(ov_model_path)

print(f"=== {model_path} ===")
print(f"{'openvino_version':<25}: {model.get_rt_info('Runtime_version').value}\n")

print_rt_info(model, ["nncf", "weight_compression"])
print_rt_info(model, "optimum")
print_rt_info(model, ["conversion_parameters", "framework"])
print(f"{'Stateful':<25}: {len(model.get_sinks()) > 0}")
