"""
Show OpenVINO device and compiled model properties.
The script shows the supported properties for all devices, and the
values of the properties for the compiled model

Prerequisites:
- an OpenVINO IR model
- `pip install openvino`

Usage: python show_compiled_model_properties.py /path/to/model_xml_or_dir

If model_xml_or_dir is a directory, it should contain a model openvino_model.xml
For models with different filenames, specify the full path to the model xml file

Optional: specify a list of devices, or an OpenVINO config as .json

Example:
python show_compiled_model_properties.py model.xml CPU GPU --config config.json
"""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import openvino as ov

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to OpenVINO model .xml file or directory containing 'openvino_model.xml'")
parser.add_argument(
    "devices",
    nargs="*",
    default=None,
    help="Optional: one or more devices to show properties for. By default properties for all available devices will be shown",
)
parser.add_argument(
    "--config",
    help="Optional: path to config.json for OpenVINO config passed to `compile_model()`. The config must be compatible with the device(s)",
)
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

ov_config = None
if args.config is not None:
    config = Path(args.config)
    if not (config.suffix == ".json" and config.is_file()):
        raise ValueError("config should point to a .json file containing an OpenVINO config.")
    with open(config) as f:
        ov_config = json.load(f)


for device in devices:
    print(f"===== {device} SUPPORTED_PROPERTIES =====")
    supported_properties = core.get_property(device, "SUPPORTED_PROPERTIES")
    for prop in sorted(supported_properties):
        if not prop == "SUPPORTED_PROPERTIES":
            try:
                value = core.get_property(device, prop)
                # read-only or read-write property
                rorw = supported_properties[prop]
            except TypeError:
                rorw = "--- error getting property ---"
            print(f"{prop} ({rorw}): {value}")
    print()

    model = core.compile_model(ov_model_path, device_name=device, config=ov_config)
    print(f"----- {ov_model_path} {device} properties -----")

    for prop in sorted(model.get_property("SUPPORTED_PROPERTIES")):
        if prop not in ["SUPPORTED_PROPERTIES", "DEVICE_PROPERTIES"]:
            try:
                value = model.get_property(prop)
            except TypeError:
                value = "--- error getting property ---"
            print(f"{prop}: {value}")
    print()

try:
    openvino_version = importlib.metadata.version("openvino")
except Exception:
    # if OpenVINO is installed from archives, importlib.metadata fails
    raise
    openvino_version = ov.__version__

if args.config is not None:
    print(f"Model compiled with OpenVINO config: {ov_config}")
print(f"OpenVINO version: {openvino_version}")
