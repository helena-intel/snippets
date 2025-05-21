"""
Show OpenVINO device and compiled model properties.
The script shows the supported properties for all devices, and the
values of the properties for the compiled model

Prerequisites:
- an OpenVINO IR model
- `pip install openvino`

Usage: python __file__ /path/to/model_xml_or_dir

If model_xml_or_dir is a directory, it should contain a model openvino_model.xml
For models with different filenames, specify the full path to the model xml file
"""

import argparse
import importlib.metadata
import sys
from pathlib import Path

import openvino as ov

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("devices", nargs="*", default=None)
args = parser.parse_args()

model_path = Path(sys.argv[1])
if model_path.is_dir() and (model_path / "openvino_model.xml").is_file():
    ov_model_path = model_path / "openvino_model.xml"
else:
    ov_model_path = model_path
    if "tokenizer.xml" in str(ov_model_path):
        import openvino_tokenizers

core = ov.Core()

devices = args.devices if args.devices else [*core.available_devices, "AUTO"]

for device in devices:
    print(f"===== {device} SUPPORTED_PROPERTIES =====")
    supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
    for prop in supported_properties:
        if not prop == "SUPPORTED_PROPERTIES":
            try:
                value = core.get_property(device, prop)
                # read-only or read-write property
                rorw = supported_properties[prop]
            except TypeError:
                rorw = "--- error getting property ---"
            print(f"{prop} ({rorw}): {value}")
    print()

    model = core.compile_model(ov_model_path, device_name=device)
    print(f"----- {ov_model_path} {device} properties -----")

    for prop in model.get_property("SUPPORTED_PROPERTIES"):
        if prop not in ["SUPPORTED_PROPERTIES", "DEVICE_PROPERTIES"]:
            try:
                value = model.get_property(prop)
            except TypeError:
                value = "--- error getting property ---"
            print(f"{prop}: {value}")
    print()

try:
    openvino_version = importlib.metadata.version("openvino")
except:
    # if OpenVINO is installed from archives, importlib.metadata fails
    raise
    openvino_version = ov.__version__

print(f"OpenVINO version: {openvino_version}")
