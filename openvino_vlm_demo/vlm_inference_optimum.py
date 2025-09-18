import time

from optimum.intel import OVModelForVisualCausalLM
from PIL import Image
from transformers import AutoProcessor, logging

logging.set_verbosity_error()

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
        self.model = None
        self.processor = None
        self.trust_remote_code = False
        self.load_model(self.device)

    def load_model(self, device):
        self.model = OVModelForVisualCausalLM.from_pretrained(
            self.model_path, device=device, trust_remote_code=self.trust_remote_code
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)

    def prepare_inputs_image(self, image, question):
        image = image.convert("RGB")
        image = resize_if_needed(image, max_size=MAX_IMAGE_SIZE)
        messages = [
            #            {"role": "system", "content": [{"type": "text", "text": "You are friendly assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image", "image": image}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        return inputs

    def run_inference_image(self, image, question):
        inputs = self.prepare_inputs_image(image, question)
        start = time.perf_counter()
        ov_output_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True,
        )
        end = time.perf_counter()
        input_length = inputs["input_ids"].shape[-1]
        ov_output_ids = ov_output_ids[0][input_length:]
        output_num_tokens = ov_output_ids.shape[0]

        text_answer = self.processor.decode(ov_output_ids, skip_special_tokens=True)
        duration = end - start
        return text_answer, output_num_tokens, duration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image")
    parser.add_argument("--model", help="path to OpenVINO VLM model directory")
    parser.add_argument("--device", choices=["GPU", "CPU", "NPU"], help="Inference device (GPU, CPU, NPU)")
    args = parser.parse_args()

    print(f"Loading {args.model} to {args.device}")
    start = time.perf_counter()
    vlm = VLM(args.model, args.device)
    end = time.perf_counter()
    print(f"Model loading completed in {end-start:.2f} seconds")

    image = Image.open(args.image)
    result_text, num_tokens, duration = vlm.run_inference_image(image, "Describe this image")
    tps, latency = perf_metrics(num_tokens, duration)
    print(result_text)
    print(f"Duration: {duration:.2f} sec, throughput: {tps} tokens/sec, latency: {latency} ms/token")
