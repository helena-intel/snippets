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

import importlib.metadata
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

core = ov.Core()

for device in [*core.available_devices, "AUTO"]:
    print(f"===== {device} SUPPORTED_PROPERTIES =====")
    supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
    for prop in supported_properties:
        if not prop == "SUPPORTED_PROPERTIES":
            value = core.get_property(device, prop)
            # read-only or read-write property
            rorw = supported_properties[prop]
            print(f"{prop} ({rorw}): {value}")
    print()

    model = core.compile_model(ov_model_path, device_name=device)
    print(f"----- {ov_model_path} {device} properties -----")

    for prop in model.get_property("SUPPORTED_PROPERTIES"):
        if prop not in ["SUPPORTED_PROPERTIES", "DEVICE_PROPERTIES"]:
            value = model.get_property(prop)
            print(f"{prop}: {value}")
    print()

try:
    openvino_version = importlib.metadata.version("openvino")
except:
    # if OpenVINO is installed from archives, importlib.metadata fails
    raise
    openvino_version = ov.__version__

print(f"OpenVINO version: {openvino_version}")
