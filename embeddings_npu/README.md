# OpenVINO embedding model inference on NPU

This is a very basic example of running embedding models with OpenVINO on NPU. The application prints the first ten elements of the model output, with L2 normalization applied.

## Prerequisites

- For model conversion and reshaping: `pip install optimum[openvino]`
- For Python inference: `pip install openvino openvino-tokenizers`
- For C++ inference: OpenVINO GenAI archive recommended, e.g. https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/2024.6/windows 

Even though we do not use OpenVINO GenAI, the GenAI archive includes openvino-tokenizers, so using that archive is the easiest way to get started with development.

## Convert the model to OpenVINO

We describe two methods to convert the model:

1. A step-by-step method, converting the model with optimum-cli, then reshaping, then adding normalization
2. An all-in-one script that converts, reshapes and normalizes the model

### Step by step model conversion


#### Convert the model to OpenVINO:

```sh
optimum-cli export openvino -m BAAI/bge-small-en-v1.5 --weight-format fp16 --task feature-extraction bge-small-en-ov
```

#### Reshape the model and tokenizer to static shapes

NPU only supports static shapes. We reshape the model to a particular shape (in this case 128) and configure the tokenizer to pad
tokens to this shape. This scripts stores the static model in bge-small-en-ov-static.

```sh
python reshape.py bge-small-en-ov
```

#### Add L2 normalization to the model

This step is optional, but very useful. Using OpenVINO's [preprocessing API](https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimize-preprocessing.html), we can embed L2 normalization in the model, so we do not have to add a normalization function to the inference script. This script stores the static model with normalization
in bge-small-en-ov-static-norm

```
python add_normalization.py bge-small-en-ov-static
```

### Convert, reshape and normalize at once

This script uses the optimum-intel API to convert and reshape the model.

```sh
python convert.py "BAAI/bge-small-en-v1.5" 
```


## Run Python inference

Run the embedding.py script with the path to the static model and the device as arguments. CPU, GPU and NPU are supported.

> [!NOTE] 
> If you did not add normalization to the model, modify the path to the model in the line below

```sh
python embedding.py bge-small-en-ov-static-norm NPU
```

## Build C++ app and run inference

In a terminal where you ran setupvars.bat or setupvars.ps1 from an extracted OpenVINO GenAI archive (see prerequisites):

```sh
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

On Windows:

```
Release\embedding.exe \path\to\model NPU
```

On Linux:

```
./embedding /path/to/model NPU
```
