import argparse
import os
from pathlib import Path

import openvino as ov
import openvino.runtime.opset15 as op
from openvino.runtime.utils.decorators import custom_preprocess_function
from openvino_tokenizers import convert_tokenizer
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoTokenizer

INPUT_SIZE = 512
BATCH_SIZE = 1

parser = argparse.ArgumentParser()
parser.add_argument("model_id")
parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
args = parser.parse_args()

args = parser.parse_args()

new_model_dir = Path(args.model_id).name + "-static-norm"
new_model_file = Path(new_model_dir) / "openvino_model.xml"

@custom_preprocess_function
def normalize(output: ov.runtime.Output):
    # stridedslice is supported on NPU, slice+squeeze is not
    output = op.strided_slice(output, begin=[0,0,0], end=[0,1,0], strides=[1,1,1], begin_mask=[1,0,0], end_mask=[1,0,1], shrink_axis_mask=[0,1,0])
    return op.normalize_l2(output, 1, 1e-12, "max")


# ### Load model and tokenizer
model = OVModelForFeatureExtraction.from_pretrained(args.model_id, export=True, compile=False, trust_remote_code=args.trust_remote_code)
hf_tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)

# Save dynamic model and tokenizers
dynamic_model_dir = str(new_model_dir).replace("-static-norm", "-dynamic")
model.save_pretrained(dynamic_model_dir)
hf_tokenizer.save_pretrained(dynamic_model_dir)
ov_tokenizer = convert_tokenizer(hf_tokenizer)
ov.save_model(ov_tokenizer, Path(dynamic_model_dir) / "openvino_tokenizer.xml")

model.reshape(BATCH_SIZE, INPUT_SIZE)

# set_layout enables using ov.set_batch() during inference to easily change the batch size of the model in runtime
for param in model.model.get_parameters():
    param.set_layout(ov.Layout("N..."))

# ### Convert tokenizer and add max padding
hf_tokenizer.model_max_length = INPUT_SIZE
ov_tokenizer = convert_tokenizer(hf_tokenizer, use_max_padding=True)

# doublecheck that tokenizer padding was added correctly
for output in ov_tokenizer.outputs:
    assert output.partial_shape[-1] == INPUT_SIZE

ov.save_model(ov_tokenizer, Path(new_model_dir) / "openvino_tokenizer.xml")

# Optional: save the model without normalization
new_model_dir_nonorm = Path(args.model_id).name + "-static"
model.save_pretrained(new_model_dir_nonorm)
ov.save_model(ov_tokenizer, Path(new_model_dir_nonorm) / "openvino_tokenizer.xml")
hf_tokenizer.save_pretrained(new_model_dir_nonorm)
model.config.save_pretrained(new_model_dir_nonorm)

# ### Add L2 normalization to model
ppp = ov.preprocess.PrePostProcessor(model.model)
ppp.output("last_hidden_state").postprocess().custom(normalize)
ppp_model = ppp.build()

# ### Save the model

os.makedirs(new_model_dir, exist_ok=True)
ov.save_model(ppp_model, new_model_file)
# hf_tokenizer and model.config are not needed for OpenVINO GenAI,
# but are needed if the model will be used with optimum-intel
hf_tokenizer.save_pretrained(new_model_dir)
model.config.save_pretrained(new_model_dir)

print(f"Model with L2 normalization and tokenizer saved to {new_model_dir}")
