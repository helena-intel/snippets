"""
Quantize a BERT sequence classification model with static post training quantization.

Documentation:
- optimum-intel quantization: https://huggingface.co/docs/optimum-intel/openvino/optimization#full-quantization
- NNCF: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html

Prerequisites:
- `pip install optimum[openvino] datasets`
- a model directory with a PyTorch BERT classification model compatible with Hugging Face's AutoModel class

Usage: `python __file__ /path/to/model_directory`
"""

import sys
from functools import partial
from pathlib import Path

from optimum.intel import OVConfig, OVModelForSequenceClassification, OVQuantizationConfig, OVQuantizer
from transformers import AutoTokenizer

model_path = Path(sys.argv[1])
print(model_path)
assert model_path.is_dir(), "model_path should point to a directory which contains a PyTorch model"
save_dir_ptq = model_path.with_name(model_path.name + "_ptq")


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", max_length=512, truncation=True)


model = OVModelForSequenceClassification.from_pretrained(model_path, export=True)
model.reshape(1, 512)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
quantizer = OVQuantizer.from_pretrained(model)

calibration_dataset = quantizer.get_calibration_dataset(
    "glue",
    dataset_config_name="sst2",
    preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
    num_samples=128,
    dataset_split="train",
)

ov_config_ptq = OVConfig(quantization_config=OVQuantizationConfig())
quantizer.quantize(ov_config=ov_config_ptq, calibration_dataset=calibration_dataset, save_directory=save_dir_ptq)
print(f"Quantized model was saved to {save_dir_ptq}")
