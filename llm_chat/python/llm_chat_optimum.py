"""This sample shows how to use OpenVINO for chat inference with optimum-intel on CPU and Intel GPU.
It is intended for testing inference outputs with optimum-intel. There is no streaming output.

Throughput and latency are displayed after inference. This is just `number of generated tokens`
divided by `generation time` (and vice versa). For a full benchmarking script that also shows
first token latency, see https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench

Prerequisites:
- pip install optimum[openvino]
- an OpenVINO LLM, exported with `optimum-cli export openvino` or downloaded from the HF hub.
  See https://github.com/helena-intel/readmes/blob/main/genai-best-practices.md

Usage: python llm_chat_optimum.py /path/to/ov_model DEVICE
"""

import argparse
import time

import torch
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.  If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

show_stats = True  # Set to False to not show inference duration/throughput/latency after every response


def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped. False means continue generation.
    return False


parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("device")
args = parser.parse_args()

# CACHE_DIR is automatically set by optimum-intel for GPU
ov_config = {}

pipe = OVModelForCausalLM.from_pretrained(args.model_dir, device=args.device, ov_config=ov_config)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

# warmup inference for GPU reproducibility
pipe.generate(input_ids=torch.as_tensor([[10]]), attention_mask=torch.as_tensor([[1]]), max_new_tokens=1, do_sample=False)

generation_config = {"max_new_tokens": 512, "do_sample": False}

messages = [{"role": "system", "content": system_prompt}]

while True:
    try:
        prompt = input("message:\n")
    except (EOFError, KeyboardInterrupt):
        break

    messages.append({"role": "user", "content": prompt})
    start = time.perf_counter()  # Include tokenization in timing
    tokenized_messages = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    tokens = tokenizer.encode_plus(tokenized_messages, add_special_tokens=False, return_tensors="pt")
    output = pipe.generate(**tokens, **generation_config)
    generated_tokens = output[0][tokens["input_ids"].shape[-1] :]
    result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    print(result_text)
    messages.append({"role": "assistant", "content": result_text})
    end = time.perf_counter()
    print("\n")
    if show_stats:
        num_tokens = generated_tokens.shape[-1]
        duration = end - start
        tps = num_tokens / duration
        avg_latency = (duration / num_tokens) * 1000
        print(
            "\033[90m"
            + f"Inference duration: {end-start:.2f} seconds. {num_tokens} tokens generated; {tps:.2f} tokens/sec; {avg_latency:.2f} ms/token."
            + "\033[0m"
        )
        print()
