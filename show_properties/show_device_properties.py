#!/usr/bin/env python
"""
Show OpenVINO device properties.
Prerequisites:
- `pip install openvino`
"""

import argparse
import importlib.metadata

import openvino as ov

core = ov.Core()

parser = argparse.ArgumentParser()
parser.add_argument("devices", nargs="*", default=None)
args = parser.parse_args()
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
                print(f"{prop} ({rorw}): {value}")
            except TypeError:
                pass
    print()


try:
    openvino_version = importlib.metadata.version("openvino")
except Exception:
    # if OpenVINO is installed from archives, importlib.metadata fails
    openvino_version = ov.__version__

print(f"OpenVINO version: {openvino_version}")
