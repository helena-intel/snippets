import subprocess
import sys
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_tokenizers
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_IDS = [
    "BAAI/bge-small-en-v1.5",
    "google-bert/bert-base-uncased",
    "distilbert/distilbert-base-uncased",
    "jinaai/jina-embeddings-v2-small-en",
    "FacebookAI/roberta-base",
]

def run_convert(model_id):
    """Runs convert.py with the given arguments and captures the output."""
    convertpy = Path(__file__).parents[1] / "convert.py"
    result = subprocess.run([sys.executable, convertpy, model_id, "--trust-remote-code"], capture_output=True, text=True)
    return result


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_1_convert_success(model_id):
    """Test that model conversion with convert.py works"""
    result = run_convert(model_id)

    assert result.returncode == 0, f"Script failed with error: {result.stderr}"
    assert "Model with L2 normalization and tokenizer saved to " in result.stdout, "Output does not match expectations"


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_2_compare_output(model_id):
    """
    Test that output of model with embedded L2 normalization matches output of PyTorch
    model with manual normalization
    """
    sentences = ["Hello world!"]
    ov_model_id = f"{Path(model_id).name}-static-norm"

    # PyTorch
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    encoded_input = tokenizer(sentences, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()

    with torch.no_grad():
        pt_model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = pt_model_output[0][:, 0]
    # normalize embeddings
    pt_sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    pt_first_embeddings = np.array(pt_sentence_embeddings[:2, :4]).round(4)

    # OpenVINO
    ov_model = ov.Core().compile_model(f"{ov_model_id}/openvino_model.xml", device_name="CPU")
    ov_tokenizer = ov.Core().compile_model(f"{ov_model_id}/openvino_tokenizer.xml", device_name="CPU")
    ov_encoded_input = ov_tokenizer(sentences)

    ov_inputs = {
        "input_ids": ov_encoded_input["input_ids"],
        "attention_mask": ov_encoded_input["attention_mask"],
    }

    if "token_type_ids" in [inp.any_name for inp in ov_model.inputs]:
        ov_inputs["token_type_ids"] = ov_encoded_input["token_type_ids"]

    ov_model_output = ov_model(ov_inputs)
    ov_sentence_embeddings = ov_model_output[0]
    ov_first_embeddings = ov_sentence_embeddings[:2, :4].round(4)
    assert np.allclose(ov_first_embeddings, pt_first_embeddings)