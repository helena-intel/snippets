#!/usr/bin/env python
"""
Example script to show model conversion properties of all the OpenVINO models in the current directory or a specified text file
If specified, the text file should have one model directory per line
For use with transformers models converted with optimum-intel

By default openvino_model.xml and openvino_language_model.xml are shown, and a subset of quantization parameters.
Modify `wc_keys` and `openvino_model_names` if needed.

Usage: python modelinfo_all.py [-f /path/to/file.txt]
Requires OpenVINO and pandas: `pip install openvino pandas`

More info: https://medium.com/openvino-toolkit/quick-checks-big-insights-debugging-openvino-models-with-rt-info-52ea86e35e95
"""

from pathlib import Path

import openvino as ov
import pandas as pd

# SETTINGS

# Pick keys to show from weight compression parameters (showing everything clutters the output, useful columns vary per use-case)
# all weight compression keys: (['all_layers', 'awq', 'backup_mode', 'compression_format', 'gptq', 'group_size', 'ignored_scope', 'lora_correction', 'mode', 'ratio', 'scale_estimation', 'sensitivity_metric'])
wc_keys = ["all_layers", "awq", "group_size", "mode", "ratio"]
openvino_model_names = ("openvino_model.xml", "openvino_language_model.xml")
strip_full_path = True  # only show model directory, not parent directories


def get_rt_info(model: ov.Model, key: str | list) -> dict:
    """
    Prints rt_info for a model for the given key. `key` can be a single key, e.g. "optimum"
    or a list, e.g. ["nncf", "weight_compression"].
    """
    d = {}
    if model.has_rt_info(key):
        rt_value = model.get_rt_info(key).value
        key = key[-1] if isinstance(key, list) else key
        if isinstance(rt_value, dict):
            for key, value in rt_value.items():
                d[key] = value
        else:
            d = {key: rt_value}
    return d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", help="optional text file with one model per line")
    args = parser.parse_args()

    if args.file is None:
        # collect modelinfo for all models in current directory (including subdirectories)
        files = (p for p in Path(".").rglob("*.xml") if p.name in openvino_model_names)
    else:
        modeldirs = [Path(line) for line in Path(args.file).read_text().splitlines()]
        files = []
        for d in modeldirs:
            files.extend(p for p in (d / fn for fn in openvino_model_names) if p.exists())

    records = []
    for model_path in files:
        model = ov.Core().read_model(model_path)
        openvino_version = model.get_rt_info("Runtime_version").value
        # make the output a bit more compact by only showing the release for OpenVINO releases - but the full version with commit otherwise
        openvino_version = (openvino_version[openvino_version.find("releases") :] if "releases" in openvino_version else openvino_version)
        model_path_str = Path(model_path).parent.name if strip_full_path else str(Path(model_path).parent)
        record = {"model": model_path_str, "openvino_version": openvino_version}
        optimum_info = record.update(get_rt_info(model, "optimum"))
        wc = get_rt_info(model, ["nncf", "weight_compression"])
        wc_record = {key: value for key, value in wc.items() if key in wc_keys}
        record.update(wc_record)
        records.append(record)

    # create and print dataframe with modelinfo
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    df = pd.DataFrame(records)
    df.fillna("", inplace=True)
    print(df)
