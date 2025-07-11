"""
OpenVINO LLM chat sample that uses greedy search for reproducible inference

Prerequisites:
- pip install openvino-genai
- an OpenVINO LLM. See https://github.com/helena-intel/readmes/blob/main/genai-best-practices.md

Usage: python llm_chat.py /path/to/ov_model DEVICE

Modified from https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/chat_sample
"""

import argparse
import time

import openvino_genai

system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.  If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

show_stats = True  # Set to False to not show inference duration after every response


def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped. False means continue generation.
    return False


parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("device")
args = parser.parse_args()

pipeline_config = {"CACHE_DIR": "model_cache"}

pipe = openvino_genai.LLMPipeline(args.model_dir, args.device, **pipeline_config)

config = pipe.get_generation_config()
config.max_new_tokens = 512
config.do_sample = False

# warmup inference for GPU reproducibility
pipe.generate("hello", max_new_tokens=1, do_sample=False)

pipe.start_chat(system_message=system_prompt)
while True:
    try:
        prompt = input("message:\n")
    except (EOFError, KeyboardInterrupt):
        break

    start = time.perf_counter()
    pipe.generate(prompt, config, streamer)
    end = time.perf_counter()
    print("\n")
    if show_stats:
        print("\033[90m" + f"Inference duration: {end-start:.2f} seconds" + "\033[0m")
        print()
pipe.finish_chat()
