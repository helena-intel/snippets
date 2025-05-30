# OpenVINO embedding model inference on NPU

This is a very basic example of running embedding models with OpenVINO on NPU, with Python and C++.
The application prints the first ten elements of the model output, with L2 normalization applied.

## Prerequisites

- For model conversion and reshaping: `pip install optimum[openvino]`
- For Python inference: `pip install openvino openvino-tokenizers`
- For C++ inference: OpenVINO GenAI archive recommended, e.g. https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.6/

Even though we do not use OpenVINO GenAI, the GenAI archive includes openvino-tokenizers, so using that archive is the easiest way to get started with development.

## Convert the model to OpenVINO

The convert.py script uses the
[optimum-intel](https://github.com/huggingface/optimum-intel) API to convert
the embedding model to OpenVINO, reshapes it to the provided shape, and adds
normalization to the model.

NPU only supports static shapes, so static shapes are required. By default we
reshape to a batch size of 1, sequence length of 128. Change BATCH_SIZE and
INPUT_SIZE in the convert.py script to change this. We also configure the
tokenizer to pad tokens to the INPUT_SIZE.

> [!NOTE]
> The batch size can be changed dynamically at inference time. To enable this,
> we add `set_layout()` to the convert.py script and `ov.set_batch()` to
> the inference code.
> It is also possible to change the sequence length dynamically, by reshaping
> the model and modifying the tokenizer. This is outside the scope of this example.

Using OpenVINO's [preprocessing
API](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html),
we can embed L2 normalization in the model. This is not required, but it is
very useful; we do not have to add a normalization function to the inference
script, and normalization is done as part of inference, also running on NPU.

This script stores the static model with normalization in {model_name}-static-norm.

To run the script, run convert.py with the model_id as argument. Feature
extraction models from the Hugging Face hub are supported by optimum-intel. Not
every model may work well on NPU. BERT models and derivates are expected to work
well and have been validated. See below for the exact models that have been tested.

```sh
python convert.py "BAAI/bge-small-en-v1.5"
```

> [!NOTE]
> For models that need remote code to be executed, add `--trust-remote-code` to `python convert.py` 

## Run Python inference

Run the embedding.py script in the `python` directory with the path to the static model and the device as
arguments. CPU, GPU and NPU are supported.

```sh
python embedding.py ..\bge-small-en-ov-static-norm NPU
```

## Build C++ app and run inference

In a terminal where you ran setupvars.bat or setupvars.ps1 from an extracted
OpenVINO GenAI archive (see prerequisites), in the `cpp` directory:

```sh
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

Run inference on Windows:

```
Release\embedding.exe ..\..\bge-small-en-ov-static-norm NPU
```

Run inference on Linux:

```
./embedding ../../bge-small-en-ov-static-norm NPU
```

## Notes

This script was tested on NPU on Ubuntu 24.10 and Windows 11 with with the following models:

-    "BAAI/bge-small-en-v1.5",
-    "google-bert/bert-base-uncased",
-    "distilbert/distilbert-base-uncased",
-    "jinaai/jina-embeddings-v2-small-en" with `--trust-remote-code`, 
-    "FacebookAI/roberta-base",

A quick test was done to check that the first couple of model outputs are close to that of the PyTorch source model, but no full validation. Use at your own risk.

