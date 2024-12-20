import openvino_tokenizers
import openvino as ov
import numpy as np
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
device = sys.argv[2]

np.set_printoptions(precision=4)

text = "Hello world!"

core = ov.Core()
ov_model = core.compile_model(model_dir / "openvino_model.xml", device)
ov_tokenizer = core.compile_model(model_dir / "openvino_tokenizer.xml", "CPU")
ov_tokens = ov_tokenizer([text,])
inputs = {"input_ids": ov_tokens["input_ids"], "attention_mask": ov_tokens["attention_mask"], "token_type_ids": ov_tokens["token_type_ids"]}
ov_result = ov_model(inputs)

print(ov_result[0].squeeze()[:10])
