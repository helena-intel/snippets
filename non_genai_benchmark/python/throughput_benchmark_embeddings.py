"""
Benchmark embedding models with OpenVINO, with asynchronous inference optimized for throughput.

Modified from https://github.com/helena-intel/snippets/tree/main/non_genai_benchmark/python which is modified from OpenVINO's
througput_benchmark.py sample https://github.com/openvinotoolkit/openvino/tree/master/samples/python/benchmark/throughput_benchmark

This version is specific for embedding models and has:
- performance counters
- logging
- percentile metrics (in log)
- custom dataset support
- configurable batch size support

Requirements: `pip install openvino openvino-tokenizers tokenizers datasets`

- This is a sample script, not intended as a production-ready benchmarking application that works out of the box for every model. You are encouraged to read
  the code and modify for your purposes!
- Use `--log logfile.csv` to log the results to an output file (one line per run). Results are APPENDED to the file if it already exists.
- Logging performance counters has a small overhead. For small models, this can be significant. Measure actual performance without performance counters,
  and then measure with performance counters to find potential performance bottlenecks
- For performance counters only one request is measured
- It is often useful to run this in a loop with logging enabled and take the median of the median latencies
  - Windows: `for /L %%i in (1,1,5) do @python throughput_benchmark_embeddings.py openvino_model.xml CPU --log log.csv`
  - Linux: `for i in {1..5}; do python throughput_benchmark_embeddings.py openvino_model.xml CPU --log log.csv; done`
- For static models, all inputs are padded to the model's input shape. For dynamic models with a batch size > 1, inputs are padded to the batch size.
  Throughput is computed based on actual unpadded input token size, including special tokens
- The model directory should contain openvino_model.xml/bin, openvino_tokenizer.xml/bin and tokenizer.json. This is the case for models exported with
  optimum-cli export openvino or https://github.com/helena-intel/snippets/blob/main/embeddings_npu/convert.py

Usage: python thoughput_benchmark.py model device
Run with `--help` to see all arguments
"""

import argparse
import csv
import datetime
import gc
import importlib.metadata
import json
import os
from pathlib import Path
from pprint import pprint
from time import perf_counter

import openvino as ov
import openvino_tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer

datasets = {
    "gsarti/flores_101": {"config": "eng", "split": "devtest", "column": "sentence"},
    "imdb": {"config": None, "split": "test", "column": "text"},
    "gsm8k": {"config": "main", "split": "test", "column": "question"},
    "BeIR/dbpedia-entity": {"config": "corpus", "split": "corpus", "column": "text"},
    "HuggingFaceFW/fineweb": {"config": "default", "split": "train", "column": "text"},
}


def list_performance_counters(request, filename=None):
    """
    Show performance counters for the request. Model must have been compiled with PERF_COUNT enabled

    Params:
    - request: inference request
    - filename: Optional filename for .csv file to store all performance counters.
    """

    profiling_list = []
    for i in request.profiling_info:
        d = {
            "node_name": i.node_name,
            "node_type": i.node_type,
            "status": i.status,
            "exec_type": i.exec_type,
            "real_time": i.real_time.microseconds,
            "cpu_time": i.cpu_time.microseconds,
        }
        profiling_list.append(d)
    profiling_list.sort(key=lambda x: x["real_time"], reverse=True)
    print()
    print(" | ".join(f"{h:<20}" for h in d.keys()))

    for row in profiling_list[:5]:
        print(" | ".join(f"{str(row[h]):<20}"[:20] for h in d.keys()))

    if filename is not None:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d.keys())
            writer.writeheader()
            for row in profiling_list:
                writer.writerow(row)


def benchmark(modelfile, device, datasetname, batch_size, num_items, user_config=None, performance_counters=False, logfile=None):
    """
    Benchmark modelfile on `device`, with asynchronous inference in THROUGHPUT mode.

    Params:
    - model: ov.Model or path to .onnx or .xml model
    - device: Inference device. CPU, GPU, NPU, AUTO
    - user_config: Optional path to json file with OpenVINO config which is passed to .compile_model
    - performance_counters: set to True to show 5 most time consuming apps after benchmark
    - logfile: Optional path to logfile. Benchmark results will be appended to this file
    """
    total = 0
    token_sizes = []

    def callback(request, userdata):
        nonlocal total
        total += 1
        token_sizes.append(userdata[-1])

    ds = datasets[datasetname]
    full_dataset = load_dataset(datasetname, ds["config"], split=f"{ds['split']}", streaming=True)
    dataset = full_dataset.take(num_items)
    sentences = [item[datasets[datasetname]["column"]] for item in dataset]
    num_sentences = len(sentences) // batch_size * batch_size
    batched_sentences = [sentences[i : i + batch_size] for i in range(0, num_sentences, batch_size)]

    core = ov.Core()

    assert isinstance(modelfile, str) or isinstance(modelfile, os.PathLike)
    model = core.read_model(Path(modelfile) / "openvino_model.xml")

    performance_hint = "THROUGHPUT" if "AUTO" not in device else "CUMULATIVE_THROUGHPUT"
    config = {"PERFORMANCE_HINT": performance_hint}
    if performance_counters:
        config["PERF_COUNT"] = performance_counters

    if user_config is not None:
        with open(user_config) as config_file:
            loaded_config = json.load(config_file)
            config.update(loaded_config)

    cpu_device = core.get_property("CPU", "FULL_DEVICE_NAME")

    if not model.is_dynamic():
        ov.set_batch(model, batch_size)
    compiled_model = core.compile_model(model, device, config)
    ov_tokenizer = core.compile_model(Path(modelfile) / "openvino_tokenizer.xml", "CPU")

    ireqs = ov.AsyncInferQueue(compiled_model)

    # Warm up
    for _ in range(len(ireqs)):
        ireqs.start_async()
    ireqs.wait_all()

    ireqs.set_callback(callback)

    # Start benchmark
    print(f"Starting benchmark on model: {modelfile}, dataset: {datasetname}, batch size: {batch_size}, sentences: {num_sentences}")
    start = perf_counter()
    for sentence in batched_sentences:
        tokens = ov_tokenizer(sentence)
        input_dict = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
        if "token_type_ids" in [item.any_name for item in model.inputs]:
            input_dict["token_type_ids"] = tokens["token_type_ids"]

        ireqs.start_async(input_dict, userdata=input_dict["input_ids"].shape)
    ireqs.wait_all()
    duration = perf_counter() - start

    # count tokens without padding
    num_non_padded_tokens = 0
    hf_tokenizer = Tokenizer.from_file(str(Path(modelfile) / "tokenizer.json"))
    # OpenVINO truncates at model max length
    hf_tokenizer.enable_truncation(max_length=max(token_sizes))
    for sentence in sentences[:num_sentences]:
        num_non_padded_tokens += len(hf_tokenizer.encode(sentence).ids)

    # Print and optionally save log
    version = importlib.metadata.version("openvino")
    d = {
        "datetime": datetime.datetime.now().isoformat(),
        "model": modelfile,
        "dataset": datasetname,
        "batch_size": batch_size,
        "num_sentences": num_sentences,
        "device": device,
        "device_name": cpu_device,
        "config": config,
        "duration": round(duration * 1000, 2),
        "total_tokens_padded": sum(token_sizes) * batch_size,
        "total_tokens_unpadded": num_non_padded_tokens,
        "throughput": round(num_non_padded_tokens / duration),
        "openvino": version,
    }
    if logfile is not None:
        writeheader = not Path(logfile).is_file()
        Path(logfile).parent.mkdir(exist_ok=True, parents=True)
        with open(logfile, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=d.keys())
            if writeheader:
                writer.writeheader()
            writer.writerow(d)

    pprint(d)

    if performance_counters:
        list_performance_counters(ireqs[0], filename=None)

    del compiled_model
    gc.collect()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model directory containing openvino_model.xml and openvino_tokenizer.xml")
    parser.add_argument("device", help="Inference device. Example: CPU, GPU, NPU")
    parser.add_argument(
        "--dataset",
        "-d",
        default="HuggingFaceFW/fineweb",
        help="Dataset to use for benchmarking. See top of script for configured datasets. Default: fineweb",
    )
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size. Default: 1")
    parser.add_argument("--num_items", "-n", type=int, default=1000, help="Max number of items from dataset. Default 1000")
    parser.add_argument(
        "--performance_counters",
        "-pc",
        action="store_true",
        help="Optional: compute performance counters and show the operations that take the most time",
    )
    parser.add_argument("--config", help="Optional: path to .json file with OpenVINO config")
    parser.add_argument(
        "--log",
        help="Optional: path to log.csv. Will be created if it does not exist. If it exists, log entries will be APPENDED to this file",
    )

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    benchmark(
        modelfile=args.model,
        device=args.device,
        datasetname=args.dataset,
        num_items=args.num_items,
        batch_size=args.batch_size,
        user_config=args.config,
        performance_counters=args.performance_counters,
        logfile=args.log,
    )
