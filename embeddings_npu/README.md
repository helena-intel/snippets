# OpenVINO embedding model inference on NPU

This is a very basic example of running embedding models with OpenVINO on NPU. The application prints the first ten elements of the model output, with L2 normalization applied.

## Prerequisites

- For model conversion and reshaping: `pip install optimum[openvino]`
- For Python inference: `pip install openvino openvino-tokenizers`
- For C++ inference: OpenVINO GenAI archive recommended, e.g. https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.6/windows

Even though we do not use OpenVINO GenAI, the GenAI archive includes openvino-tokenizers, so using that archive is the easiest way to get started with development.

## Convert the model to OpenVINO

The convert.py script uses the
[optimum-intel](https://github.com/huggingface/optimum-intel) API to convert
the embedding model to OpenVINO, reshapes it to the provided shape, and adds
normalization to the model.

NPU only supports static shapes, so static shapes are required. By default we
reshape to 384, change SIZE in the convert.py script to change this. We also
configure the tokenizer to pad tokens to this shape.

Using OpenVINO's [preprocessing
API](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html),
we can embed L2 normalization in the model. This is not required, but it is
very useful; we do not have to add a normalization function to the inference
script, and normalization is done as part of inference, also running on NPU.

This script stores the static model with normalization in {model_name}-static-norm.

To run the script, run convert.py with the model_id as argument. Feature
extraction models from the Hugging Face hub are supported by optimum-intel. Not
every model may work well on NPU. BERT models and derivates are expected to work
well and have been validated, and experimentally BAAI/bge works well too. For example:

```sh
python convert.py "BAAI/bge-small-en-v1.5"
```

## Run Python inference

Run the embedding.py script with the path to the static model and the device as
arguments. CPU, GPU and NPU are supported.

```sh
python embedding.py bge-small-en-ov-static-norm NPU
```

## Build C++ app and run inference

In a terminal where you ran setupvars.bat or setupvars.ps1 from an extracted
OpenVINO GenAI archive (see prerequisites):

```sh
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Run inference on Windows:

```
Release\embedding.exe \path\to\model NPU
```

Run inference on Linux:

```
./embedding /path/to/model NPU
```
