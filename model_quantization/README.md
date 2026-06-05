# Model Quantization

Simple static post-training quantization example for BERT models.

## Scripts

- **quantize_bert_classification.py** — Quantize a BERT sequence classification model using NNCF static quantization with the GLUE SST-2 calibration dataset.

## Requirements

```bash
pip install optimum-intel datasets
```

## Usage 

```
python quantize_bert_classification.py /path/to/pytorch_model_directory
```

The `pytorch_model_directory` should contain a PyTorch BERT classification model, in Hugging Face format, for example from [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=text-classification&sort=trending).

The quantized model is saved to `<model_directory>_ptq`.

## References

- [Optimum Intel quantization docs](https://huggingface.co/docs/optimum-intel/openvino/optimization#full-quantization)
- [NNCF post-training quantization](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html)
