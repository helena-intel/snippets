# OpenVINO Benchmarking Scripts

This directory contains a collection of Python scripts and shell scripts for benchmarking OpenVINO models across different devices and configurations.

The benchmarking script are based on [OpenVINO samples](https://github.com/openvinotoolkit/openvino/tree/master/samples/python/benchmark).

## Scripts Overview

### Core Benchmarking Scripts

- **`sync_benchmark.py`** - Synchronous inference benchmark optimized for latency measurement
- **`throughput_benchmark.py`** - Asynchronous inference benchmark optimized for throughput measurement  
- **`throughput_benchmark_embeddings.py`** - Specialized throughput benchmark for embedding models with real dataset support
- **`sync_benchmark_hetero.py`** - Extension of sync benchmark using HETERO plugin for mixed device inference

### Automation Scripts

Example files that demonstrate how to run inference in a loop and capture system information

- **`benchmark.bat`** - Windows batch script for automated benchmarking with system info logging
- **`benchmark.sh`** - Linux/macOS bash script for automated benchmarking loops

### Analysis Script

- **`pivot_benchmark.py`** - Example script to show median performance per model/device

## Key Features

- **Multiple inference modes**: Synchronous (latency-focused) and asynchronous (throughput-focused)
- **Device support**: CPU, GPU, NPU, AUTO, and HETERO configurations
- **Performance analysis**: Built-in performance counters and percentile metrics
- **Logging**: CSV output for result tracking and analysis
- **Model flexibility**: Support for static and dynamic models with on-the-fly reshaping
- **Real datasets**: Embedding benchmark supports HuggingFace datasets
- **System integration**: Automated system info collection and OpenVINO property logging

## Requirements

```bash
pip install openvino
```

Additional requirements for specific scripts:
- Embedding benchmark: `pip install openvino-tokenizers tokenizers datasets`
- Pivot analysis: `pip install pandas`

## Usage Examples

### Basic Benchmarking

```bash
# Latency benchmark
python sync_benchmark.py model.xml CPU --log results.csv

# Throughput benchmark
python throughput_benchmark.py model.xml GPU --log results.csv

# Embedding model benchmark
python throughput_benchmark_embeddings.py model_dir NPU --dataset imdb --batch_size 4
```

### Automated Benchmarking

```bash
# Windows
benchmark.bat model.xml logfile

# Linux/macOS
source benchmark.sh
```

### Performance Analysis

Show the five most time-consuming operations

```bash
# View performance counters
python sync_benchmark.py model.xml CPU --performance_counters
```

### Pivot analysis (median per model/device)

Use `pivot_benchmark.py` to summarize CSV logs into a compact table of medians. The script expects columns `model`, `device`, and the metric column (default: `throughput`).

```bash
# Median throughput per model with devices as columns
python pivot_benchmark.py logtest.csv

# Filter to a single device and output a one-column table of medians
python pivot_benchmark.py logtest.csv -d GPU.0

# Filter models by substring and save the pivot to CSV
python pivot_benchmark.py logtest.csv -m bge -c throughput -o pivot_out.csv
```

## Output

All benchmarks generate detailed CSV logs with metrics including:
- Latency percentiles (median, p90, p95, p99)
- Throughput measurements
- Device and system information
- OpenVINO configuration details
- Timestamp and model information

Results are appended to log files, making it easy to track performance across multiple runs and compare different configurations.
