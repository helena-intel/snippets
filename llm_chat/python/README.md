# LLM Chat

Interactive LLM chat samples using OpenVINO GenAI

Scripts:
- **llm_chat.py** — Interactive chat with streaming output and greedy decoding
- **llm_chat_manual.py** — Chat with manual tokenization control
- **llm_chat_optimum.py** — Chat using Optimum Intel pipeline
- **llm_test.py** — Chat sample for models/tokenizers that do not have a supported chat template. Does not keep chat history, intended for testing models only.


## Requirements

```bash
pip install openvino-genai
```

## Usage

```
python python/llm_chat.py /path/to/ov_model DEVICE
```

- `path/to/ov_model` should point to a directory with an LLM in OpenVINO format. Convert your own model with [optimum-intel](https://huggingface.co/docs/optimum-intel/openvino/export) or download a model from Hugging Face Hub, for example from [OpenVINO's collection](https://huggingface.co/OpenVINO) or [LLMWare's Model Depot](https://huggingface.co/collections/llmware/model-depot).
- `DEVICE` should be a supported device on your system. `CPU` should usually work, on Intel AI PCs you can use `GPU` and `NPU` too.

