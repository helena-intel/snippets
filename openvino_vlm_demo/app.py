"""
Gradio app for Visual AI inference with OpenVINO

Usage: python app.py [model] [device]
"""

import argparse

import gradio as gr

from vlm_inference_genai import VLM as VLM_GENAI
from vlm_inference_optimum import VLM as VLM_OPTIMUM
from vlm_inference_ovms import VLM as VLM_OVMS

css = """
.text textarea {font-size: 24px !important;}
"""


def perf_metrics(num_tokens, duration):
    tps = round(num_tokens / duration, 2)
    latency = round((duration / num_tokens) * 1000, 2)
    return tps, latency


def process_inputs(image, question):
    if image is None:
        return "Please upload an image."
    if not question:
        return "Please enter a question."
    result_text, num_tokens, duration = vlm.run_inference_image(image, question)
    tps, latency = perf_metrics(num_tokens, duration)
    print(f"Duration: {duration:.2f} sec, throughput: {tps} tokens/sec, latency: {latency} ms/token")
    return result_text


def reset_inputs():
    return None, "", ""


def launch_demo():
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# OpenVINO Visual Language Model (VLM) Demo")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload an Image", height=300, width=500)
            with gr.Column():
                text_input = gr.Textbox(label="Enter a Question", elem_classes="text")
                output_text = gr.Markdown(label="Answer", height=400)

        with gr.Row():
            process_button = gr.Button("Process")
            reset_button = gr.Button("Reset")

        gr.Examples(
            [["Describe the image"]],
            text_input,
        )

        process_button.click(process_inputs, inputs=[image_input, text_input], outputs=output_text)
        text_input.submit(process_inputs, inputs=[image_input, text_input], outputs=output_text)
        reset_button.click(reset_inputs, inputs=[], outputs=[image_input, text_input, output_text])
        demo.launch(server_port=7790)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="app.py", description="Gradio demo for VLMs with OpenVINO using Openvino GenAI, OVMS or Optimum Intel"
    )
    subparsers = parser.add_subparsers(dest="type", required=True, help="Mode to run")

    p_genai = subparsers.add_parser("genai", help="Run with OpenVINO GenAI")
    p_genai.add_argument("model_path", help="Path to the local model")
    p_genai.add_argument("device", help="Device to use ('CPU', 'GPU', 'NPU')")

    p_ovms = subparsers.add_parser("ovms", help="Run with OVMS (expects running OVMS server)")

    p_optimum = subparsers.add_parser("optimum", help="Run with OpenVINO's integration in transformers, Optimum Intel")
    p_optimum.add_argument("model_path", help="Path to the local model")
    p_optimum.add_argument("device", help="Device to use ('CPU', 'GPU', 'NPU')")

    return parser.parse_args()


args = parse_args()

if args.type == "genai":
    vlm = VLM_GENAI(args.model_path, args.device)
elif args.type == "ovms":
    vlm = VLM_OVMS()
elif args.type == "optimum":
    vlm = VLM_OPTIMUM(args.model_path, args.device)

launch_demo()
