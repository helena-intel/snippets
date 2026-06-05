# Model Info

Inspect metadata and quantization properties of OpenVINO models. Useful for verifying which quantization settings and framework versions were used during model conversion, and comparing configurations across multiple models.

## Scripts

- **modelinfo.py** — Show conversion and quantization info for a single model (framework versions, NNCF weight compression config, Optimum Intel config, stateful status).
- **modelinfo_all.py** — Show a comparison table of quantization properties for all models in a directory or from a list file.

The framework versions in the output, for OpenVINO, NNCF, Transformers etc., are the versions that were used for converting the model to OpenVINO.

## Requirements

```bash
pip install openvino
```

`modelinfo_all.py` additionally requires `pandas`:

```bash
pip install pandas
```

## Usage 

### Single model

```
python modelinfo.py /path/to/model_dir_or_xml [--show_model]
```

- `--show_model` shows the model inputs and outputs, including shape and data type.
- `model_dir_or_xml` can point to a directory if that directory contains `openvino_model.xml`. For models with other names, use the full path to the model's .xml file.

#### All models in current directory

```
python modelinfo_all.py
```

#### Models from a file (one path per line)

```
python modelinfo_all.py -f models.txt
```

## Sample Output

### modelinfo.py

```bash
=== c:\users\helena\models\llama-3.2-3b-instruct-ov-int4 ===
openvino_version         : 2025.4.0-20381-2120be664d3-releases/2025/4

advanced_parameters      : {'statistics_path': None, 'lora_adapter_rank': 256, 'group_size_fallback_mode': 'ignore', 'min_adjusted_group_size': 16, 'awq_params': {'subset_size': 32, 'percent_to_apply': 0.002, 'alpha_min': 0.0, 'alpha_max': 1.0, 'steps': 100, 'prefer_data_aware_scaling': True}, 'scale_estimation_params': {'subset_size': 64, 'initial_steps': 5, 'scale_steps': 5, 'weight_penalty': -1.0}, 'gptq_params': {'damp_percent': 0.1, 'block_size': 128, 'subset_size': 128}, 'lora_correction_params': {'adapter_rank': 8, 'num_iterations': 3, 'apply_regularization': True, 'subset_size': 128, 'use_int8_adapters': True}, 'backend_params': {}, 'codebook': None}
all_layers               : False
awq                      : False
backup_mode              : int8_asym
compression_format       : dequantize
gptq                     : False
group_size               : 128
ignored_scope            : []
lora_correction          : False
mode                     : int4_asym
ratio                    : 1.0
scale_estimation         : False
sensitivity_metric       : weight_quantization_error

nncf_version             : 2.18.0
optimum_intel_version    : 1.26.1
optimum_version          : 2.0.0
pytorch_version          : 2.9.0
transformers_version     : 4.55.4

framework                : pytorch
Stateful                 : True
```

### modelinfo_all.py

```bash
                               model openvino_version nncf_version optimum_intel_version optimum_version pytorch_version transformers_version all_layers    awq group_size       mode ratio
0      llama-3.2-1b-instruct-ov-int4  releases/2025/4       2.18.0                1.26.1           2.0.0           2.9.0               4.55.4      False   True        128  int4_asym   1.0
1  llama-3.2-1b-instruct-ov-int4-sym  releases/2026/2        3.1.0   1.27.0.dev0+a8c4734      2.1.0.dev0      2.12.0+cpu                5.0.0      False  False        128   int4_sym   1.0
2      llama-3.2-3b-instruct-ov-int4  releases/2025/4       2.18.0                1.26.1           2.0.0           2.9.0               4.55.4      False  False        128  int4_asym   1.0
3    granite-3.3-8b-instruct-ov-int4  releases/2025/4       2.18.0                1.26.1           2.0.0           2.9.0               4.55.4      False  False        128  int4_asym   1.0
```

# See also

[Quick Checks, Big Insights: Debugging OpenVINO Models with rt_info](https://medium.com/openvino-toolkit/quick-checks-big-insights-debugging-openvino-models-with-rt-info-52ea86e35e95)
