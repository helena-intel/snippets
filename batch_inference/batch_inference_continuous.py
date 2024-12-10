import csv
import importlib.metadata
import subprocess
import sys
import time

import openvino_genai as ov_genai
from datasets import load_dataset

# SETTINGS
num_items = 512 # Number of dataset items. 
max_new_tokens = 50
scheduler_cache_size = 16  # amount of memory to use for scheduler, in GB. Set this to a higher value on Xeon.
datasetname = "gsarti/flores_101"  # See datasets line below for configured datasets

datasets = {
    "gsarti/flores_101": {"config": "eng", "split": "devtest", "column": "sentence"},
    "imdb": {"config": None, "split": "test", "column": "text"},
    "gsm8k": {"config": "main", "split": "test", "column": "question"},
}

if len(sys.argv) != 3:
    raise ValueError(f"Usage: python {__file__} MODEL_PATH MAX_BATCH_TOKENS")

model_path = sys.argv[1]
max_num_batched_tokens = int(sys.argv[2])

# set up model and tokenizer
scheduler = ov_genai.SchedulerConfig()
scheduler.cache_size = scheduler_cache_size
# max_num_batched_tokens is 256 by default, for acceptable latency
# To improve throughput, set max_num_batched_tokens to a higher value
scheduler.max_num_batched_tokens = max_num_batched_tokens

# pipe = ov_genai.ContinuousBatchingPipeline(model_path, scheduler, "CPU", {"EXECUTION_MODE_HINT":"ACCURACY"})
pipe = ov_genai.ContinuousBatchingPipeline(model_path, scheduler, "CPU")
tokenizer = pipe.get_tokenizer()

config = pipe.get_config()
config.max_new_tokens = max_new_tokens
config.do_sample = False
config.ignore_eos = True

# warmup
wconfig = ov_genai.GenerationConfig()
wconfig.max_new_tokens = 2
pipe.generate(["hello world"], [wconfig])


# prepare dataset
ds = datasets[datasetname]
dataset = load_dataset(datasetname, ds["config"], split=f"{ds['split']}[0:{num_items}]")
sentences = [item[datasets[datasetname]["column"]] for item in dataset]

# inference
configs = [
    config,
] * len(sentences)
start_time = time.time()
# Line below works too instead of manually tokenizing
# output = pipe.generate(sentences, configs)
input_tokens = [tokenizer.encode(prompt).input_ids for prompt in sentences]
result = pipe.generate(input_tokens, configs)
output = [tokenizer.decode(item.m_generation_ids[0]) for item in result]

end_time = time.time()
total_duration = end_time - start_time

num_tokens = sum(len(item.m_generation_ids[0]) for item in result)
assert num_tokens == max_new_tokens * len(sentences)
throughput = num_tokens / total_duration

print(f"Total generated tokens: {num_tokens}")
print(f"Total duration with {num_items} dataset items, dataset {datasetname}: {total_duration:.2f} seconds")
print(f"{throughput:.2f} tok/sec, {throughput*60:.2f} tok/min")
print(f"do_sample: {config.do_sample}")
print(f"OpenVINO Genai {importlib.metadata.version('openvino_genai')}")
