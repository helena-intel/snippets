# Evaluate OpenVINO embedding models with MTEB
# See https://github.com/embeddings-benchmark/mteb
#
# Dependencies: `pip install mteb sentence_transformers optimum[openvino]`

import os
from pathlib import Path

import mteb
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model_name can be a model_id from the Hugging Face Hub (model will be converted to OpenVINO automatically)
# or a path to a local model (without embedded normalization)
model_name = "BAAI/bge-small-en-v1.5"
model_name = "/home/ubuntu/repos/snippets/embeddings_npu/bge-small-en-v1.5-static"

model = SentenceTransformer(model_name, backend="openvino")
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=f"mteb_results/{Path(model_name).name}")
print(results[0].scores)
