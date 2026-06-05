# OpenVINO C++ chat sample

Interactive LLM chat sample using OpenVINO GenAI

## Prepare environment:

Download and uncompress an [OpenVINO archive](https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/).
setupvars.bat|ps1|sh are in the root of the uncompressed archive file.

### Windows

Run this in a Developer Command Prompt / PowerShell

```shell
\path\to\setupvars.bat|ps1
```

### Linux

```shell
source /path/to/setupvars.sh
```

## Build:

```shell
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Run 

> [!IMPORTANT]  
> Run/source setupvars before running the application

Usage:

```
llm_chat /path/to/ov_model device
```

- `path/to/ov_model` should point to a directory with an LLM in OpenVINO format. Convert your own model with [optimum-intel](https://huggingface.co/docs/optimum-intel/openvino/export) or download a model from Hugging Face Hub, for example from [OpenVINO's collection](https://huggingface.co/OpenVINO) or [LLMWare's Model Depot](https://huggingface.co/collections/llmware/model-depot).
- `device` should be a supported device on your system. `CPU` should usually work, on Intel AI PCs you can use `GPU` and `NPU` too.


### Windows example:

```
Release\llm_chat.exe \path\to\model GPU
```

### Linux example:

```
./llm_chat /path/to/model GPU
```

## Deploy

To deploy this application including necessary DLLs, see these [CMakeLists](https://github.com/helena-intel/snippets/tree/main/genai_cmakelists).
