"""
Add L2 normalization to OpenVINO embedding model exported with `optimum-cli export openvino --task feature-extraction`

Usage: python add_normalization.py /path/to/model
The model with normalization included will be stored in {original_model}-norm.
"""

import os
import shutil
import sys
from pathlib import Path

import openvino as ov
import openvino.runtime.opset15 as op
from openvino.runtime.utils.decorators import custom_preprocess_function

MODEL_DIR = sys.argv[1]
MODEL_FILE = "openvino_model.xml"

model_path = Path(MODEL_DIR) / MODEL_FILE
ppp_model_dir = f"{MODEL_DIR}-norm"
ppp_model_file = Path(ppp_model_dir) / MODEL_FILE


@custom_preprocess_function
def normalize(output: ov.runtime.Output):
    output = op.squeeze(output)
    output = op.slice(output, [0, ], [1, ], [1, ])
    return op.normalize_l2(output, 1, 1e-12, "max")


core = ov.Core()

model = core.read_model(model_path)
ppp = ov.preprocess.PrePostProcessor(model)
ppp.output("last_hidden_state").postprocess().custom(normalize)
ppp_model = ppp.build()
os.makedirs(ppp_model_dir, exist_ok=True)
ov.save_model(ppp_model, ppp_model_file)
shutil.copy(Path(MODEL_DIR) / "openvino_tokenizer.xml", ppp_model_dir)
shutil.copy(Path(MODEL_DIR) / "openvino_tokenizer.bin", ppp_model_dir)

print(f"Model with L2 normalization and tokenizer saved to {ppp_model_dir}")
