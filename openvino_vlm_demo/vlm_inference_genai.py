"""
Example script to show VLM inference with OpenVINO GenAI.
The script can be used standalone, or as part of the Gradio demo at
https://github.com/helena-intel/snippets/tree/main/openvino_vlm_demo

Images are resized to MAX_IMAGE_SIZE before inference.

Usage: python vlm_inference_genai.py <image> --model <model_path> --device <CPU|GPU|NPU> [--prompt <text>]

Requirements: `pip install openvino-genai pillow`
"""

import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino_genai import VLMPipeline
from PIL import Image

MAX_IMAGE_SIZE = 512


def resize_if_needed(image: Image.Image, max_size: int) -> Image.Image:
    if image.width >= max_size or image.height >= max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def perf_metrics(num_tokens, duration):
    tps = round(num_tokens / duration, 2)
    latency = round((duration / num_tokens) * 1000, 2)
    return tps, latency


class VLM:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.load_model(self.device)

    def load_model(self, device):
        pipeline_config = {"CACHE_DIR": str(Path(self.model_path) / "model_cache")}
        self.model = VLMPipeline(self.model_path, device, **pipeline_config)

    def prepare_inputs_image(self, image):
        image = image.convert("RGB")  # transparency is not supported, this is very fast if image is already 3 channels
        image = resize_if_needed(image, max_size=MAX_IMAGE_SIZE)
        image_array = np.array(image.getdata())
        image_data = image_array.reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
        image_data = ov.Tensor(image_data)
        return image_data

    def run_inference_image(self, image, question):
        image_data = self.prepare_inputs_image(image)
        start = time.perf_counter()
        result = self.model.generate(question, image=image_data, do_sample=False, max_new_tokens=512)
        end = time.perf_counter()
        num_tokens = result.perf_metrics.get_num_generated_tokens()
        duration = end - start
        return result.texts[0], num_tokens, duration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image")
    parser.add_argument("--model", help="path to OpenVINO VLM model directory")
    parser.add_argument("--device", choices=["GPU", "CPU", "NPU"], help="Inference device (GPU, CPU, NPU)")
    parser.add_argument("--prompt", help="Prompt. Default: 'Describe this image'", default="Describe this image")
    args = parser.parse_args()

    print(f"Loading {args.model} to {args.device}")
    start = time.perf_counter()
    vlm = VLM(args.model, args.device)
    end = time.perf_counter()
    print(f"Model loading completed in {end-start:.2f} seconds")

    image = Image.open(args.image)
    result_text, num_tokens, duration = vlm.run_inference_image(image, args.prompt)
    result_text, num_tokens, duration = vlm.run_inference_image(image, args.prompt)
    tps, latency = perf_metrics(num_tokens, duration)
    print(result_text)
    print(f"Duration: {duration:.2f} sec, throughput: {tps} tokens/sec, latency: {latency} ms/token")
