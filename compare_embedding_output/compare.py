"""
Visually compare embedding model output between:

- PyTorch model
- OpenVINO model with Optimum Intel
- Base OpenVINO with Hugging Face tokenizers
- Base OpenVINO with OpenVINO tokenizers
- Base OpenVINO with embedding included in model, with OpenVINO tokenizers

Paths to OpenVINO models are defined at the top of the script and must exist
before running the script.
See https://github.com/helena-intel/snippets/blob/main/embeddings_npu/convert.py
for a model conversion script.
"""

pt_model_id = "BAAI/bge-small-en-v1.5"
ov_model_id = "bge-small-en-v1.5-static"
ov_model_id_norm = "bge-small-en-v1.5-static-norm"

SIZE = 384  # padding size, should be same as ov_model size if static models are used.

# If model is exported with fixed batch size, sentences should be same length as batch size
sentences = ["Hello world!", ]
# sentences = ["Hello world!", "OpenVINO prepreprocessing is very cool!"]

import numpy as np
import openvino as ov
import torch
from optimum.intel import OVModelForFeatureExtraction
from transformers import AutoModel, AutoTokenizer
import openvino_tokenizers


tokenizer = AutoTokenizer.from_pretrained(pt_model_id)
encoded_input = tokenizer(sentences, padding="max_length", max_length=SIZE, truncation=True, return_tensors="pt")

#### PyTorch

model = AutoModel.from_pretrained(pt_model_id)
model.eval()

# Compute token embeddings
with torch.no_grad():
    pt_model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = pt_model_output[0][:, 0]
# normalize embeddings
pt_sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
print("PyTorch                                :", np.array(pt_sentence_embeddings[:2, :4]).round(4))

#### OpenVINO Optimum

model = OVModelForFeatureExtraction.from_pretrained(ov_model_id)

# Compute token embeddings
opt_model_output = model(**encoded_input)
# Perform pooling. In this case, cls pooling.
sentence_embeddings = opt_model_output[0][:, 0]
# normalize embeddings
opt_sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

print("OpenVINO Optimum                       :", np.array(opt_sentence_embeddings[:2, :4]).round(4))

#### OpenVINO Base with Hugging Face tokenizer

device = "CPU"  # choose CPU or GPU
ov_config = {"CACHE_DIR": "model_cache"} if device == "GPU" else None
ov_model = ov.Core().compile_model(f"{ov_model_id}/openvino_model.xml", device_name=device, config=ov_config)


def normalize(x, p, axis, eps=1e-12):
    norm = np.linalg.norm(x, ord=p, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)

if "token_type_ids" not in encoded_input:
    encoded_input["token_type_ids"] = np.zeros_like(encoded_input["attention_mask"])

ov_inputs = {
    "input_ids": encoded_input.input_ids,
    "attention_mask": encoded_input.attention_mask,
    "token_type_ids": encoded_input.token_type_ids,
}
ov_model_output = ov_model(ov_inputs)
sentence_embeddings = ov_model_output[0][:, 0]
ov_sentence_embeddings = normalize(sentence_embeddings, p=2, axis=1)

print("OpenVINO + HF Tokenizers               :", ov_sentence_embeddings[:2, :4].round(4))

#### OpenVINO Base with OpenVINO tokenizer

ov_tokenizer = ov.Core().compile_model(f"{ov_model_id}/openvino_tokenizer.xml", device_name="CPU")
ov_encoded_input = ov_tokenizer(sentences)

ov_inputs2 = {
    "input_ids": ov_encoded_input["input_ids"],
    "attention_mask": ov_encoded_input["attention_mask"],
    "token_type_ids": ov_encoded_input["token_type_ids"],
}
ov_model_output2 = ov_model(ov_inputs2)
sentence_embeddings = ov_model_output2[0][:, 0]
ov_sentence_embeddings2 = normalize(sentence_embeddings, p=2, axis=1)

print("OpenVINO + OpenVINO Tokenizers         :", ov_sentence_embeddings2[:2, :4].round(4))

#### OpenVINO Base with embedded normalizion, with OpenVINO tokenizer

ov_model = ov.Core().compile_model(f"{ov_model_id_norm}/openvino_model.xml", device_name=device, config=ov_config)
ov_tokenizer = ov.Core().compile_model(f"{ov_model_id_norm}/openvino_tokenizer.xml", device_name="CPU")
ov_encoded_input = ov_tokenizer(sentences)

ov_inputs3 = {
    "input_ids": ov_encoded_input["input_ids"],
    "attention_mask": ov_encoded_input["attention_mask"],
    "token_type_ids": ov_encoded_input["token_type_ids"],
}
ov_model_output3 = ov_model(ov_inputs3)
ov_sentence_embeddings3 = ov_model_output3[0]

print("OpenVINO with L2 + OpenVINO Tokenizers :", ov_sentence_embeddings3[:2, :4].round(4))
