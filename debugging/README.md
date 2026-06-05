# Debugging

Performance debugging script for OpenVINO Vision Language models (VLMs).

## Scripts

- **vision_embedding_performance.py** — Profile the vision embeddings model (part of a VLM) using OpenVINO performance counters. Shows the top 5 most time-consuming operations.

Tested with InternVL, SmolVLM, LLaVA, MiniCPM-V-2_6, gemma-3. Does not work with Qwen or Phi models.

## Usage

```bash
python vision_embedding_performance.py /path/to/vlm_model_dir GPU
python vision_embedding_performance.py /path/to/vlm_model_dir CPU --log counters.csv
```

## Requirements

```
pip install openvino
```

## Sample Output

```
python vision_embedding_performance.py gemma-3-4b-it-ov-int4 GPU
Input shape: [(1, 3, 896, 896)]
Inference 1 duration (GPU): 1.01 seconds

node_name                                                                                                                         | node_type | exec_type           | real_time | cpu_time | percentage_time
__module.model.vision_tower.vision_model.encoder.layers.13.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention | SDPA      | ocl::sdpa::opt__f16 | 19062     | 4        | 2.75
__module.model.vision_tower.vision_model.encoder.layers.19.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention | SDPA      | ocl::sdpa::opt__f16 | 19028     | 3        | 2.75
__module.model.vision_tower.vision_model.encoder.layers.2.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention  | SDPA      | ocl::sdpa::opt__f16 | 18947     | 6        | 2.74
__module.model.vision_tower.vision_model.encoder.layers.9.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention  | SDPA      | ocl::sdpa::opt__f16 | 18672     | 22       | 2.7
__module.model.vision_tower.vision_model.encoder.layers.25.self_attn/aten::scaled_dot_product_attention/ScaledDotProductAttention | SDPA      | ocl::sdpa::opt__f16 | 18640     | 3        | 2.69
```
