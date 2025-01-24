import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_tokenizers
from datasets import load_dataset

# SETTINGS
num_items = 512  # Number of dataset items.
datasetname = "gsarti/flores_101"  # See datasets line below for configured datasets
datasetname = "imdb"  # See datasets line below for configured datasets
device = "CPU"

datasets = {
    "gsarti/flores_101": {"config": "eng", "split": "devtest", "column": "sentence"},
    "imdb": {"config": None, "split": "test", "column": "text"},
    "gsm8k": {"config": "main", "split": "test", "column": "question"},
}

if len(sys.argv) != 3:
    raise ValueError(f"Usage: python {__file__} MODEL_PATH BATCH_SIZE")

model_path = Path(sys.argv[1])
batch_size = int(sys.argv[2])

# set up model and tokenizer
core = ov.Core()
ov_model = core.read_model(model_path / "openvino_model.xml")
ov.set_batch(ov_model, batch_size)
ov_compiled_model = core.compile_model(ov_model, device)
ov_tokenizer = core.compile_model(model_path / "openvino_tokenizer.xml", "CPU")

warmup_input = ov_tokenizer(["hello"] * batch_size)
warmup_input_dict = {input_name.any_name: input_data for (input_name, input_data) in warmup_input.to_dict().items()}
ov_compiled_model(warmup_input_dict)

# prepare dataset
ds = datasets[datasetname]
dataset = load_dataset(datasetname, ds["config"], split=f"{ds['split']}[0:{num_items}]")
sentences = [item[datasets[datasetname]["column"]] for item in dataset]
batched_sentences = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]


def do_inference(prompt):
    tick = time.perf_counter()
    ov_tokens = ov_tokenizer(prompt)
    inputs = {input_name.any_name: input_data for (input_name, input_data) in ov_tokens.to_dict().items()}
    result = ov_compiled_model(inputs)
    tock = time.perf_counter()
    duration = tock - tick
    return duration


durations = []
start_time = time.time()
for prompt in batched_sentences:
    if len(prompt) != batch_size:
        prompt += ["hello"] * (batch_size - len(prompt))
    duration = do_inference(prompt)
    durations.append(duration * 1000)  # capture duration in ms
end_time = time.time()

total_duration = end_time - start_time

print(f"{num_items} dataset items, dataset {datasetname}, batch size {batch_size} (batches: {len(durations)})")
print(f"Total duration: {total_duration:.2f} seconds, batch duration (ms): mean: {np.mean(durations):.1f}, median: {np.median(durations):.1f}, max: {np.max(durations):.1f}")
