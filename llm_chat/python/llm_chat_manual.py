"""
This sample shows how to use OpenVINO GenAI for chat inference with a manual chat history,
instead of using `pipe.start_chat()` and `pipe.finish_chat()`.

Throughput and latency are displayed after inference. This is just `number of generated tokens`
divided by `generation time` (and vice versa). For a full benchmarking script that also shows
first token latency, see https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench

Prerequisites:
- pip install openvino-genai
- an OpenVINO LLM. See https://github.com/helena-intel/readmes/blob/main/genai-best-practices.md

Usage: python llm_chat_manual.py /path/to/ov_model DEVICE
"""

import argparse
import time

import openvino_genai

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

pipeline_config = {"CACHE_DIR": "model_cache"}

pipe = openvino_genai.LLMPipeline(args.model_dir, args.device, **pipeline_config)
ov_tokenizer = pipe.get_tokenizer()

config = pipe.get_generation_config()
config.max_new_tokens = 512
config.do_sample = False

if openvino_genai.__version__ >= "2025.1":
    config.apply_chat_template = False

# warmup inference for GPU reproducibility
pipe.generate("hello", max_new_tokens=1, do_sample=False)

messages = [{"role": "system", "content": system_prompt}]

while True:
    try:
        prompt = input("message:\n")
    except (EOFError, KeyboardInterrupt):
        break

    messages.append({"role": "user", "content": prompt})
    start = time.perf_counter()  # Include tokenization in timing
    tokenized_messages = ov_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    tokens = ov_tokenizer.encode(tokenized_messages, add_special_tokens=False)
    result = pipe.generate(tokens, config, streamer)
    result_text = ov_tokenizer.decode(result.tokens[0])
    messages.append({"role": "assistant", "content": result_text})
    end = time.perf_counter()
    print("\n")
    if show_stats:
        num_tokens = len(result.tokens[0])
        duration = end - start
        tps = num_tokens / duration
        avg_latency = (duration / num_tokens) * 1000
        print(
            "\033[90m"
            + f"Inference duration: {end-start:.2f} seconds. {num_tokens} tokens generated; {tps:.2f} tokens/sec; {avg_latency:.2f} ms/token."
            + "\033[0m"
        )
        print()
