"""
Example script for VLM inference with OVMS.
OVMS server is expected to be running on base_url, defined at the top of the script.
OVMS instructions:
- General: https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_vlm.html
- NPU: https://docs.openvino.ai/2025/model-server/ovms_demos_vlm_npu.html
Error checking is out of scope for this script.

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
base_url = "http://localhost:8000/v3"


def resize_if_needed(image: Image.Image, max_size: int) -> Image.Image:
    if image.width >= max_size or image.height >= max_size:
        image = image.copy()
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image


def perf_metrics(num_tokens, duration):
    tps = round(num_tokens / duration, 2)
    latency = round((duration / num_tokens) * 1000, 2)
    return tps, latency


def convert_image(image_file: str | os.PathLike):
    """
    Convert image from image-file to base64
    """
    with open(image_file, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image


def pil_to_data_url(image: Image.Image):
    buf = BytesIO()
    image.save(buf, "PNG")
    b = buf.getvalue()
    return f"data:image/png;base64,{base64.b64encode(b).decode('ascii')}"


class VLM:
    def __init__(self):
        ovms_config = requests.get(f"{base_url.replace('v3', 'v1')}/config").json()
        print(ovms_config)
        self.model_name = list(ovms_config)[0]

    def run_inference_image(self, image: Image.Image, question: str):
        image = resize_if_needed(image, max_size=MAX_IMAGE_SIZE)
        data_url = pil_to_data_url(image)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_completion_tokens": 512,
            "temperature": 0
        }
        headers = {"Content-Type": "application/json", "Authorization": "not used"}
        start = time.perf_counter()
        response = requests.post(base_url + "/chat/completions", json=payload, headers=headers)
        end = time.perf_counter()
        duration = end - start

        try:
            result = json.loads(response.text)
            result_text = result["choices"][0]["message"]["content"]
            num_tokens = result["usage"]["completion_tokens"]
        except KeyError:
            result = f"An error occured: {response.text}"
        return result_text, num_tokens, duration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image")
    args = parser.parse_args()
    vlm = VLM()
    image = Image.open(args.image)
    result_text, num_tokens, duration = vlm.run_inference_image(image, "Describe this image")
    tps, latency = perf_metrics(num_tokens, duration)
    print(result_text)
    print(f"Duration: {duration:.2f} sec, throughput: {tps} tokens/sec, latency: {latency} ms/token")
