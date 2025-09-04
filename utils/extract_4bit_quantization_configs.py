"""
Extract default 4-bit weight quantization configs from
https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/configuration.py

These are the default configs that are used when exporting a transformers model to OpenVINO with
`optimum-cli export openvino --weight-format int4`
without specifying any additional parameters (like `--sym` or `--ratio`).

Requirements: `pip install optimum[openvino]`
"""

import pandas as pd
from optimum.intel.openvino.configuration import _DEFAULT_4BIT_WQ_CONFIGS

markdown_filename = "default_openvino_int4_configs.md"

df = pd.DataFrame.from_records(_DEFAULT_4BIT_WQ_CONFIGS).T
df.fillna("", inplace=True)

# only keep 4-bit weight only quantization configs
to_drop = df[df["quantization_configs"] != ""].index
df.drop(to_drop, inplace=True)
df.drop(["bits", "quantization_configs"], inplace=True, axis=1)

df["quant_method"] = df["quant_method"].apply(lambda x: "" if x == "" else x.name)
df.index.name = "model"
df.sort_index(key=lambda s: s.str.casefold(), inplace=True)

start_text = """# Default OpenVINO quantization configs for INT4 quantization

This page lists the default quantization configurations for OpenVINO LLMs. These configs are used when using `optimum-cli` with
`--weight-format int4` without specifying additional quantization parameters:

```
optimum-cli export openvino -m model_id --weight-format int4 model-id-ov-int4
```

The table shows settings that are explicitly set. If a cell is empty, the setting is not set. For example, for the
`HuggingFaceH4/zephyr-7b-beta` model, `AWQ` is set, but `scale_estimation` is not, and no dataset is specified for AWQ (data-free AWQ will
be used). The default config for this model is equivalent to doing

```
optimum-cli export openvino -m HuggingFaceH4/zephyr-7b-beta --sym --group-size 128 --ratio 0.8 --awq zephyr-7b-beta-ov-int4
```

"""

end_text = """

[Source](https://github.com/huggingface/optimum-intel/blob/main/optimum/intel/openvino/configuration.py#L54)

[`optimum-cli export openvino` documentation](https://huggingface.co/docs/optimum/main/en/intel/openvino/export)
"""

with open(markdown_filename, "w") as f:
    f.write(start_text)
    f.write(df.to_markdown(index=True))
    f.write(end_text)
