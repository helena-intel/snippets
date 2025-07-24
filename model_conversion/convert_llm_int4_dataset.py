"""
This snippets shows how to export a text-generation model to OpenVINO INT4 with a custom dataset
Quantization statistics are optionally cached for faster quantization when quantizing multiple times

Requirements: `pip install optimum[openvino]`

Usage: modify settings in # Settings section and run the script
"""

import os
import re
from pathlib import Path

import openvino as ov
from datasets import load_dataset
from huggingface_hub import snapshot_download
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
from optimum.intel.openvino.configuration import _DEFAULT_4BIT_WQ_CONFIG, _DEFAULT_4BIT_WQ_CONFIGS
from transformers import AutoTokenizer

# Settings
dataset_name = "allenai/c4"
dataset_subset = "zh"
dataset_split = "train"
dataset_column = "text"
model_id = "microsoft/Phi-4-mini-instruct"
npu_support = False  # set to True to enable symmetric channel-wise quantization
save_dir = "phi-4-mini-instruct-ov-int4-awq-zh"
trust_remote_code = False


print(f"*** Exporting {model_id} with {dataset_name}[{dataset_subset}] ({dataset_split})")

# Download the model (if not already downloaded) and get the cache directory
snapshot_dir = snapshot_download(model_id)

# Load, convert and save tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "False"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
tokenizer.save_pretrained(save_dir)
ov_tokenizers = convert_tokenizer(tokenizer, with_detokenizer=True)
ov.save_model(ov_tokenizers[0], Path(save_dir) / "openvino_tokenizer.xml")
ov.save_model(ov_tokenizers[1], Path(save_dir) / "openvino_detokenizer.xml")

# Load dataset
ds = load_dataset(dataset_name, dataset_subset, split=dataset_split, streaming=True)
texts = [item[dataset_column] for item in ds.take(300)]

# Create quantization config, based on default config in optimum-intel if available
wq_config = _DEFAULT_4BIT_WQ_CONFIGS.get(model_id, _DEFAULT_4BIT_WQ_CONFIG)
wq_config.pop("dataset", None)
if npu_support:
    # Override defaults with NPU friendly settings. This may impact accuracy.
    wq_config["group_size"] = -1
    wq_config["sym"] = True
    wq_config["ratio"] = 1.0
print("*** Quantization config:", wq_config)
# Save statistics for faster quantization after the first time
# statistics_dir = re.sub(r"[^a-z0-9_]", "_", f"statistics_{dataset_name}_{dataset_subset}_{dataset_split}")
# advanced_parameters = AdvancedCompressionParameters(statistics_path=Path(snapshot_dir) / statistics_dir)
# wq_config["advanced_parameters"] = advanced_parameters
quantization_config = OVWeightQuantizationConfig(**wq_config, dataset=texts)

# Export model with quantization config and save exported model
model = OVModelForCausalLM.from_pretrained(
    model_id, export=True, compile=False, quantization_config=quantization_config, trust_remote_code=trust_remote_code
)
model.save_pretrained(str(save_dir))
print(f"*** Exported model and tokenizers were saved to {save_dir}")
