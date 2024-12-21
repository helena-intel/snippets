import argparse
import os
from pathlib import Path

import openvino as ov
import openvino.runtime.opset15 as op
from openvino.runtime.utils.decorators import custom_preprocess_function
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

SIZE = 384

parser = argparse.ArgumentParser()
parser.add_argument("model_id")
args = parser.parse_args()

new_model_dir = Path(args.model_id).name + "-static-norm"
new_model_file = Path(new_model_dir) / "openvino_model.xml"

@custom_preprocess_function
def normalize(output: ov.runtime.Output):
    output = op.slice(output, [0,], [1], [1,], [1])
    output = op.squeeze(output, axes=[1,])
    return op.normalize_l2(output, 1, 1e-12, "max")


# ### Export and reshape the model
model = OVModelForFeatureExtraction.from_pretrained(args.model_id, export=True, compile=False)
model.reshape(-1, SIZE)

# ### Convert tokenizer and add max padding
hf_tokenizer = AutoTokenizer.from_pretrained(args.model_id)
hf_tokenizer.model_max_length = SIZE
ov_tokenizer = convert_tokenizer(hf_tokenizer, use_max_padding=True)

# doublecheck that tokenizer padding was added correctly
for output in ov_tokenizer.outputs:
    assert output.partial_shape[-1] == SIZE

ov.save_model(ov_tokenizer, Path(new_model_dir) / "openvino_tokenizer.xml")

# Optional: uncomment the lines below to save the model without normalization
new_model_dir_nonorm = Path(args.model_id).name + "-static"
model.save_pretrained(new_model_dir_nonorm)
ov.save_model(ov_tokenizer, Path(new_model_dir_nonorm) / "openvino_tokenizer.xml")

# ### Add L2 normalization to model
ppp = ov.preprocess.PrePostProcessor(model.model)
ppp.output("last_hidden_state").postprocess().custom(normalize)
ppp_model = ppp.build()
os.makedirs(new_model_dir, exist_ok=True)
ov.save_model(ppp_model, new_model_file)

print(f"Model with L2 normalization and tokenizer saved to {new_model_dir}")
