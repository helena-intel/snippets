import openvino_tokenizers
import openvino as ov
import numpy as np
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
device = sys.argv[2]

np.set_printoptions(precision=4)

prompts = ["Hello world!", "OpenVINO is great"]

core = ov.Core()
ov_model = core.read_model(model_dir / "openvino_model.xml")
ov.set_batch(ov_model, len(prompts))
ov_compiled_model = core.compile_model(ov_model, device)
ov_tokenizer = core.compile_model(model_dir / "openvino_tokenizer.xml", "CPU")
ov_tokens = ov_tokenizer(prompts)
inputs = {input_name.any_name: input_data for (input_name, input_data) in ov_tokens.to_dict().items()}
ov_result = ov_compiled_model(inputs)

print(ov_result[0][:,:10])
