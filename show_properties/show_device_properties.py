"""
Show OpenVINO device properties.
Prerequisites:
- `pip install openvino`
"""

import importlib.metadata

import openvino as ov

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


try:
    openvino_version = importlib.metadata.version("openvino")
except:
    # if OpenVINO is installed from archives, importlib.metadata fails
    openvino_version = ov.__version__

print(f"OpenVINO version: {openvino_version}")
