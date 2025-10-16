"""
List models supported by optimum-intel that are unsupported by the latest version of transformers.

Background: Some transformers models with remote code have not been updated to support the latest transformers releases.
To convert these models to OpenVINO, a version of transformers that supports the model must be installed. This script
lists the maximum transformers version for these models.  Only models that are not supported with the latest version
are listed - models that are not listed in the output of this script are supported with the latest transformers version
supported by optimum-intel.

Optimum Intel release notes (includes supported transformer versions for each release): https://github.com/huggingface/optimum-intel/releases/
Optimum Intel supported models: https://huggingface.co/docs/optimum/main/intel/openvino/models

Requirements for running the script: `pip install optimum[openvino]`
"""

import inspect

import pandas as pd
from optimum.exporters.openvino import model_configs

versions = {}

for name, obj in inspect.getmembers(model_configs):
    if "OpenVINOConfig" in name and inspect.isclass(obj):
        max_version = getattr(obj, "MAX_TRANSFORMERS_VERSION", None)
        if max_version is not None:
            versions[name.replace("OpenVINOConfig", "")] = max_version

df = pd.DataFrame.from_records([versions]).T
df.columns = ["Max transformers version"]
print(df.sort_index())
