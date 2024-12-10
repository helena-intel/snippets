"""
Show OpenVINO device and compiled model properties.
The script shows the supported properties for all devices, and the 
values of the properties for the compiled model

Prerequisites:
- an OpenVINO IR model
- `pip install openvino`

Usage: python __file__ /path/to/model
"""

import importlib.metadata
import sys

import openvino as ov

# sys.argv[1] is expected to be a directory containing openvino_model.xml
# To use another model name, modify the compile_model line below
model_directory = sys.argv[1]

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

    model = core.compile_model(f"{model_directory}/openvino_model.xml", device_name=device)
    print(f"----- {model_directory} {device} properties -----")

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
