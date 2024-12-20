import os
import sys
from pathlib import Path

import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer

SIZE = 128

model_dir = Path(sys.argv[1])

core = ov.Core()

model = core.read_model(model_dir / "openvino_model.xml")
model.reshape({"input_ids": (1, SIZE), "attention_mask": (1, SIZE), "token_type_ids": (1, SIZE)})
print(model)
new_dir = model_dir.name + "-static"
os.makedirs(new_dir, exist_ok=True)
ov.save_model(model, Path(new_dir) / "openvino_model.xml")

hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
hf_tokenizer.model_max_length = SIZE
ov_tokenizer = convert_tokenizer(hf_tokenizer, use_max_padding=True)
ov.save_model(ov_tokenizer, Path(new_dir) / "openvino_tokenizer.xml")
print(ov_tokenizer)

print(f"model and tokenizer saved to {new_dir}")
