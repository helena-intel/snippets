# OpenVINO Visual Language Model demo

> [!CAUTION]
> work in progress!

This directory contains a [gradio](https://www.gradio.app) demo for running inference on visual language models
with [OpenVINO](https://github.com/openvinotoolkit/openvino) using OpenVINO, supporting:

- [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server/)
- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
- [Optimum Intel](https://github.com/huggingface/optimum-intel)

This is example code to build upon, not intended to be production ready
inference code. This is also why there is overlap between the inference files -
they are also meant to be standalone examples of running inference on these
models. The vlm_inference_transformers.py code is included for easy comparison
with the source model. It is not added to the Gradio demo.

## Usage

### Install dependencies

The demo requires `gradio` (`pip install gradio`) and the inference package of choice.

- OpenVINO GenAI: `pip install openvino-genai`
- Optimum Intel: `pip install optimum[openvino]`
- OpenVINO Model Server: [OVMS VLM documentation](https://docs.openvino.ai/2025/model-server/ovms_demos_continuous_batching_vlm.html) (Specific for NPU: [OVMS VLM on NPU documentation](https://docs.openvino.ai/2025/model-server/ovms_demos_vlm_npu.html))


### Export or download a model

For a quick start, download a pre-converted model, after `pip install huggingface-hub`:

```
huggingface-cli download helenai/Qwen2.5-VL-3B-Instruct-ov-int4 --local-dir Qwen2.5-VL-3B-Instruct-ov-int4
```

Or for an NPU-friendly model:

```
huggingface-cli download helenai/Qwen2.5-VL-3B-Instruct-ov-int4-npu --local-dir Qwen2.5-VL-3B-Instruct-ov-int4-npu
```

Or export your own model:

- [Export model to OpenVINO documentation](https://huggingface.co/docs/optimum/main/en/intel/openvino/export)
- [List of Supported VLM models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/#visual-language-models-vlms) (not all of these models are supported on NPU, see list in the link below)
- [Export model for NPU](https://github.com/helena-intel/readmes/blob/main/openvino_llm_model_export_npu.md)

### Run the Gradio demo

Use GPU if you have a recent Intel laptop with iGPU or an Intel discrete GPU.
If you have an Intel AI PC, you can try NPU. Otherwise use CPU. On GPU and NPU, it may
take a while to compile the model. This will be faster on subsequent runs.


Using OpenVINO GenAI:

```
python app.py genai --model Qwen2.5-VL-3B-Instruct-ov-int4 --device GPU
```

Using Optimum Intel:

```
python app.py optimum --model Qwen2.5-VL-3B-Instruct-ov-int4 --device GPU
```

Using OVMS requires a running OVMS server. See documentation linked above.

```
python app.py genai ovms
```


The output from app.py will show `Running on local URL:  http://127.0.0.1:7790`. Click on this link to open the demo in your browser (or type in this URL manually).

### Run inference in a script

The inference scripts allow you to run inference on a particular image. Edit the script to modify the prompt. The script shows the model output and throughput/latency.

```
python vlm_inference_genai.py --model  Qwen2.5-VL-3B-Instruct-ov-int4 --device GPU image.jpg
```

