#!/usr/bin/env python
"""
Example script to show model conversion properties of OpenVINO LLM models converted with optimum-intel.
Shows basic info for other OpenVINO models.

Usage: python modelinfo.py /path/to/model_directory_or_xml_file
Requires OpenVINO: `pip install openvino`

If a model_directory is given, the model file is assumed to be openvino_model.xml (this is the default for LLMs).
For other models, pass the full path to the model's xml file.

More info: https://medium.com/openvino-toolkit/quick-checks-big-insights-debugging-openvino-models-with-rt-info-52ea86e35e95
"""

from pathlib import Path

import openvino as ov


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


def show_model_info(model_path, show_model=False):
    print(f"=== {model_path} ===")
    model_path = Path(model_path)
    if model_path.is_dir() and (model_path / "openvino_model.xml").is_file():
        ov_model_path = model_path / "openvino_model.xml"
    elif model_path.is_file():
        ov_model_path = model_path
    else:
        raise ValueError(
            "model_path should point to a model file (for example model.xml), or a model directory containing openvino_model.xml"
        )

    model = ov.Core().read_model(ov_model_path)
    if model.has_rt_info("Runtime_version"):
        print(f"{'openvino_version':<25}: {model.get_rt_info('Runtime_version').value}\n")
    print_rt_info(model, ["nncf", "weight_compression"])
    print_rt_info(model, "optimum")
    print_rt_info(model, ["conversion_parameters", "framework"])
    print(f"{'Stateful':<25}: {len(model.get_sinks()) > 0}")
    if show_model:
        print()
        print(model)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model file (.xml, .onnx etc) or directory containing openvino_model.xml")
    parser.add_argument("--show_model", action="store_true", help="Show model inputs and outputs")
    args = parser.parse_args()
    show_model_info(args.model_path, args.show_model)
