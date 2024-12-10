import csv
import importlib.metadata
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openvino_genai as ov_genai
from datasets import load_dataset

# SETTINGS
num_items = 512 # Number of dataset items. I set this to 64 for smaller batch sizes and 256/1024 for higher, to speed up testing.
max_new_tokens = 50
datasetname = "gsarti/flores_101"  # See datasets line below for configured datasets

datasets = {
    "gsarti/flores_101": {"config": "eng", "split": "devtest", "column": "sentence"},
    "imdb": {"config": None, "split": "test", "column": "text"},
    "gsm8k": {"config": "main", "split": "test", "column": "question"},
}

if len(sys.argv) != 3:
    raise ValueError(f"Usage: python {__file__} MODEL_PATH BATCH_SIZE")

model_path = sys.argv[1]
batch_size = int(sys.argv[2])

# set up model and tokenizer
pipe = ov_genai.LLMPipeline(model_path, "CPU")
tokenizer = pipe.get_tokenizer()

config = pipe.get_generation_config()
config.max_new_tokens = max_new_tokens
config.do_sample = False
config.ignore_eos = True

# warmup
pipe.generate(["hello world"] * batch_size, max_new_tokens=2)

# prepare dataset
ds = datasets[datasetname]
dataset = load_dataset(datasetname, ds["config"], split=f"{ds['split']}[0:{num_items}]")
sentences = [item[datasets[datasetname]["column"]] for item in dataset]
batched_sentences = [sentences[i : i + batch_size] for i in range(0, len(sentences), batch_size)]


def do_inference(prompt):
    tick = time.perf_counter()
    # line below works too, but with explicitly tokenizing we can count generated tokens
    # outputs = pipe.generate(prompt, generation_config=config)
    input_tokens = tokenizer.encode(prompt)
    result = pipe.generate(input_tokens, generation_config=config)
    output = tokenizer.decode(result.tokens)
    tock = time.perf_counter()
    duration = tock - tick
    num_tokens = sum([len(item) for item in result.tokens])
    return output, duration, num_tokens


durations = []
num_tokens_list = []
results = []
start_time = time.time()
for prompt in batched_sentences:
    result, duration, num_token = do_inference(prompt)
    durations.append(duration)
    results += result
    num_tokens_list.append(num_token)
end_time = time.time()

num_tokens = sum(num_tokens_list)
assert num_tokens == max_new_tokens * len(sentences)

total_duration = end_time - start_time
throughput = num_tokens / total_duration

print(f"Total generated tokens: {num_tokens}")
print(f"Total duration with {num_items} dataset items, dataset {datasetname}, batch size {batch_size}: {total_duration:.2f} seconds")
print(f"{throughput:.2f} tok/sec, {throughput*60:.2f} tok/min")
print(f"do_sample: {config.do_sample}")
print(f"OpenVINO Genai {importlib.metadata.version('openvino_genai')}")
