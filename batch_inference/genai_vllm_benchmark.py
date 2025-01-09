# For vLLM+OpenVINO installation, follow https://docs.vllm.ai/en/latest/getting_started/installation/openvino.html
#
# Requirements (apart from vLLM): `pip install openvino-genai datasets`

import argparse
import csv
import os
import time

import cpuinfo
import openvino_genai as ov_genai
from datasets import load_dataset
from vllm import LLM, SamplingParams

# SETTINGS
max_num_batched_tokens = 2000  # for GenAI

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", required=True)
parser.add_argument("-f", "--framework", choices=("vllm", "genai"), required=True)
parser.add_argument("-n", "--num_items", "--num-items", type=int, required=True)
parser.add_argument("--memory_size", "--memory-size", type=int, required=True, help="amount of memory to use in GB.")
parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=50)
parser.add_argument("--dataset", choices=("flores_101", "imdb", "gsm8k"), default="flores_101")
parser.add_argument("--logfile", help="Optional path to log file. csv data is appended to the file")
args = parser.parse_args()

if args.framework == "vllm":
    os.environ["VLLM_OPENVINO_KVCACHE_SPACE"] = str(args.memory_size)
    pipe = LLM(model=args.model_path)
    config = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, ignore_eos=True)
else:
    scheduler = ov_genai.SchedulerConfig()
    scheduler.cache_size = args.memory_size
    # max_num_batched_tokens is 256 by default, for acceptable latency
    # To improve throughput, set max_num_batched_tokens to a higher value
    scheduler.max_num_batched_tokens = max_num_batched_tokens
    pipe = ov_genai.ContinuousBatchingPipeline(args.model_path, scheduler, "CPU")
    config = pipe.get_config()
    config.max_new_tokens = args.max_new_tokens
    config.do_sample = False
    config.ignore_eos = True


datasets = {
    "gsarti/flores_101": {"config": "eng", "split": "devtest", "column": "sentence"},
    "imdb": {"config": None, "split": "test", "column": "text"},
    "gsm8k": {"config": "main", "split": "test", "column": "question"},
}


# prepare dataset
datasetname = "gsarti/flores_101" if args.dataset == "flores_101" else args.dataset
ds = datasets[datasetname]
dataset = load_dataset(datasetname, ds["config"], split=f"{ds['split']}[0:{args.num_items}]")
sentences = [item[datasets[datasetname]["column"]] for item in dataset]

# inference
start_time = time.time()
result = pipe.generate(sentences, [config] * len(sentences))
end_time = time.time()
total_duration = end_time - start_time

results = args.__dict__
results["cpu"] = cpuinfo.get_cpu_info().get("brand_raw")
results["duration"] = round(total_duration, 2)

for key, value in results.items():
    print(f"{key:<20}: {value}")

if args.logfile is not None:
    with open(args.logfile, "a+") as f:
        results.pop("logfile")
        d = csv.DictWriter(f, fieldnames=results.keys())
        if f.tell() == 0:
            d.writeheader()
        else:
            f.seek(0)
        d.writerow(results)
