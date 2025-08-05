"""
Download subset of dataset from Hugging Face Hub to a json file.

Requirements: `pip install datasets`

Usage: modify datasetname and limit at the top of the script and run the script
"""

import json
import re
from pathlib import Path

from datasets import load_dataset

datasetname = "HuggingFaceFW/fineweb"
limit = 1000  # Number of dataset items to save

datasetstring = re.sub(r"(?:[^\w]|_)+", "_", datasetname)  # replace non-word characters with underscore
datasetfile = f"{datasetstring}_{limit}.jsonl"
if not Path(datasetfile).is_file():
    full_dataset = load_dataset(datasetname, split="train", streaming=True)
    subset = full_dataset.take(limit)

    with open(datasetfile, "w", encoding="utf-8") as f:
        for example in subset:
            f.write(json.dumps(example, ensure_ascii=False))
            f.write("\n")
    print(f"Wrote dataset to {datasetfile}")
else:
    print(f"{datasetfile} already exists")

print(f'Load dataset with: `dataset = load_dataset("json", data_files="{datasetfile}")["train"]`')
