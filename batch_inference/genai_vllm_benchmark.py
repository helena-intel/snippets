# For vLLM+OpenVINO installation, follow https://docs.vllm.ai/en/latest/getting_started/installation/openvino.html
#
# Requirements (apart from vLLM): `pip install openvino-genai datasets`

import argparse
import csv
import os
import time

import cpuinfo
import openvino_genai as ov_genai
import vllm
from datasets import load_dataset
from transformers import AutoConfig
from vllm.config import _get_and_verify_max_len

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", required=True)
parser.add_argument("-f", "--framework", choices=("vllm", "genai"), required=True)
parser.add_argument("-n", "--num_items", "--num-items", type=int, required=True)
parser.add_argument("--memory_size", "--memory-size", type=int, required=True, help="amount of memory to use in GB.")
parser.add_argument("--max_new_tokens", "--max-new-tokens", type=int, default=50)
parser.add_argument("--dataset", choices=("flores_101", "imdb", "gsm8k", "custom"), default="flores_101")
parser.add_argument("--logfile", help="Optional path to log file. csv data is appended to the file")
args = parser.parse_args()

# Align max_num_batched_tokens with vLLM default
model_config = AutoConfig.from_pretrained(args.model_path)
max_num_batched_tokens = _get_and_verify_max_len(model_config, max_model_len=None, disable_sliding_window=True, sliding_window_len=None)


if args.framework == "vllm":
    os.environ["VLLM_OPENVINO_KVCACHE_SPACE"] = str(args.memory_size)
    pipe = vllm.LLM(model=args.model_path, max_num_batched_tokens=max_num_batched_tokens)
    config = vllm.SamplingParams(temperature=0, max_tokens=args.max_new_tokens, ignore_eos=True)
else:
    scheduler = ov_genai.SchedulerConfig()
    scheduler.cache_size = args.memory_size
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
if args.dataset == "custom":
    # modify this for custom dataset loading
    import pandas as pd

    df = pd.read_excel("questions.xlsx")
    args.num_items = args.num_items if args.num_items > -1 else len(df)
    sentences = df["question"].to_list()[: args.num_items]
else:
    datasetname = "gsarti/flores_101" if args.dataset == "flores_101" else args.dataset
    ds = datasets[datasetname]
    dataset = load_dataset(datasetname, ds["config"], split=f"{ds['split']}[0:{args.num_items}]")
    sentences = [item[datasets[datasetname]["column"]] for item in dataset]

# warmup
pipe.generate(["hello"], [config])

# inference
start_time = time.time()
result = pipe.generate(sentences, [config] * len(sentences))
end_time = time.time()
total_duration = end_time - start_time

results = args.__dict__
results["max_batched_tokens"] = max_num_batched_tokens
results["framework_version"] = ov_genai.__version__ if args.framework == "genai" else vllm.__version__
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
