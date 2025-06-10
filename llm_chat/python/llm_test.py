"""
OpenVINO LLM chat sample without chat template. This sample is meant to test models that do not have a chat
template. For better results, use a chat model (usually named -instruct or -chat) and use the
llm_chat.py sample instead. This chat will not have history, it is purely meant to test model outputs.

Prerequisites:
- pip install openvino-genai
- an OpenVINO LLM.

Usage: python llm_test.py /path/to/ov_model DEVICE

Modified from https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/chat_sample
"""

import argparse
import time

import openvino_genai


def streamer(subword):
    print(subword, end="", flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False


parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("device")
args = parser.parse_args()

pipeline_config = {"CACHE_DIR": "model_cache"}

pipe = openvino_genai.LLMPipeline(args.model_dir, args.device, **pipeline_config)

config = pipe.get_generation_config()
config.max_new_tokens = 100
config.do_sample = False
config.apply_chat_template = False  # From 2025.1, chat templates are automatically enabled if they exist for the model

# warmup inference
pipe.generate("hello", max_new_tokens=1, do_sample=False, apply_chat_template=False)

while True:
    try:
        prompt = input("prompt:\n")
    except EOFError:
        break

    start = time.perf_counter()
    pipe.generate(prompt, config, streamer)
    end = time.perf_counter()
    print()
    print(f"Inference duration: {end-start:.2f} seconds")
    print("\n----------")
