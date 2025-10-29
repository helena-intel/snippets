"""
Example script for VLM inference with OVMS.
OVMS server is expected to be running on base_url, assumed to be localhost:port/v3. Modify in `if __name__` part if needed.

The script can be used standalone, or as part of the Gradio demo at
https://github.com/helena-intel/snippets/tree/main/openvino_vlm_demo

OVMS instructions:
- General: https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_vlm.html
- NPU: https://docs.openvino.ai/2025/model-server/ovms_demos_vlm_npu.html

Images are resized to MAX_IMAGE_SIZE before inference.
Error checking is out of scope for this script.

Usage: python vlm_inference_ovms.py <image> [--ovms_port <port>] [--prompt <text>]

Requirements: Python 3.10 or higher, `pip install pillow requests`
"""

import base64
import json
import os
import time
from io import BytesIO

import requests
from PIL import Image

MAX_IMAGE_SIZE = 512
MAX_COMPLETION_TOKENS = 512


def resize_if_needed(image: Image.Image, max_size: int) -> Image.Image:
    if image.width >= max_size or image.height >= max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def convert_image(image_file: str | os.PathLike):
    """
    Convert image from image-file to base64
    """
    with open(image_file, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image


def pil_to_data_url(image: Image.Image, fmt="JPEG"):
    buf = BytesIO()
    image.save(buf, fmt)
    b = buf.getvalue()
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(b).decode('ascii')}"


def perf_metrics(num_tokens, duration):
    """
    Compute tokens/sec and ms/token. Returns (0, 0) when num_tokens == 0. If duration <= 0, TPS is 0.
    """
    if num_tokens == 0:
        return 0, 0
    tps = round(num_tokens / duration, 2) if duration > 0 else 0
    latency = round((duration / num_tokens) * 1000, 2)
    return tps, latency


class VLM:
    def __init__(self, base_url):
        self.base_url = base_url
        ovms_config = requests.get(f"{self.base_url.replace('v3', 'v1')}/config").json()
        self.model_name = list(ovms_config)[0]

    def run_inference_image(self, image: Image.Image, question: str):
        image = image.convert("RGB")  # transparency is not supported, this is very fast if image is already 3 channels
        image = resize_if_needed(image, max_size=MAX_IMAGE_SIZE)
        data_url = pil_to_data_url(image, fmt="PNG")
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "temperature": 0,
            "top_p": 1,
        }
        headers = {"Content-Type": "application/json", "Authorization": "not used"}
        start = time.perf_counter()
        response = requests.post(self.base_url + "/chat/completions", json=payload, headers=headers)
        end = time.perf_counter()
        duration = end - start

        try:
            result = json.loads(response.text)
            result_text = result["choices"][0]["message"]["content"]
            num_tokens = result["usage"]["completion_tokens"]
        except Exception:
            result_text = f"An error occured: {response.text}"
            num_tokens = 0
        return result_text, num_tokens, duration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image")
    parser.add_argument("--ovms_port", "-p", type=int, default=8000)
    parser.add_argument("--prompt", help="Prompt. Default: 'Describe this image'", default="Describe this image")
    args = parser.parse_args()
    base_url = f"http://localhost:{args.ovms_port}/v3"
    vlm = VLM(base_url)
    image = Image.open(args.image)
    result_text, num_tokens, duration = vlm.run_inference_image(image, args.prompt)
    tps, latency = perf_metrics(num_tokens, duration)
    print(result_text)
    print(f"Duration: {duration:.2f} sec, throughput: {tps} tokens/sec, latency: {latency} ms/token")
