# Loading Encrypted Models with OpenVINO

This snippet demonstrates how to load an AES-128-CBC encrypted OpenVINO IR model entirely in memory, without writing decrypted files to disk.

The approach and code are based on Fiona Zhao's blog [Use Encrypted Model with OpenVINO](https://blog.openvino.ai/blog-posts/use-encrypted-model-with-openvino).

## Background

OpenVINO supports loading models from in-memory buffers via `core.read_model(xml_bytes, bin_bytes)`. This is the recommended approach for encrypted models: decrypt into memory and pass the buffers directly. No decrypted files are written to disk, and OpenVINO never opens a file handle on the decrypted data.

## Preparation

### 1. Download a model

You can use any OpenVINO IR model. For this example we use a small text detection model.

```bash
curl -o model.xml https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.3/models_bin/1/horizontal-text-detection-0001/FP16-INT8/horizontal-text-detection-0001.xml
curl -o model.bin https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.3/models_bin/1/horizontal-text-detection-0001/FP16-INT8/horizontal-text-detection-0001.bin
```

### 2. Encrypt the model

This requires the `openssl` executable. On Linux, install it with for example `sudo apt install openssl`, on Windows
`openssl` is bundled with Git for Windows.

```bash
openssl enc -aes-128-cbc -in model.xml -out model_enc.xml -K 6f70656e76696e6f20656e6372797074 -iv 6f70656e76696e6f20656e6372797074
openssl enc -aes-128-cbc -in model.bin -out model_enc.bin -K 6f70656e76696e6f20656e6372797074 -iv 6f70656e76696e6f20656e6372797074
```

For simplicity, in this example the key and IV are both `6f70656e76696e6f20656e6372797074` (the hex encoding of "openvino encrypt").

## Python

### Requirements

```bash
pip install openvino cryptography
```

### Run

```bash
python infer_encrypted.py model_enc.xml --key 6f70656e76696e6f20656e6372797074 --iv 6f70656e76696e6f20656e6372797074
```

This decrypts the model in memory, compiles it, and prints input/output shapes. If the model has static shapes with floating point inputs, dummy inference is also run.

## C++

### Requirements

- CMake 3.10+
- OpenVINO 2023.0+ (https://docs.openvino.ai/2026/get-started/install-openvino.html, recommended: install from archives)
- OpenSSL development libraries:
  - **Linux**: `sudo apt install libssl-dev` (Debian/Ubuntu) or `sudo dnf install openssl-devel` (Fedora)
  - **Windows**: Install from https://slproweb.com/products/Win32OpenSSL.html (choose "Win64 OpenSSL v3.x", full version, not "Light"), or use another method (see https://github.com/openssl/openssl/blob/master/INSTALL.md). The `openssl` CLI from Git for Windows is not sufficient for building C++ code.

### Build

Run this in a terminal after setting OpenVINO environment variables with setupvars.bat/.ps1/.sh

```bash
cmake -B build
cmake --build build --config Release
```

On Windows you may need to point CMake at your dependencies:

```bash
cmake -B build -DOpenVINO_DIR=<path_to_openvino_cmake> -DOPENSSL_ROOT_DIR=<path_to_openssl>
```

### Run

```bash
# Linux
./build/model_crypto model_enc.xml 6f70656e76696e6f20656e6372797074 6f70656e76696e6f20656e6372797074

# Windows
build\Release\model_crypto.exe model_enc.xml 6f70656e76696e6f20656e6372797074 6f70656e76696e6f20656e6372797074
```

An optional fourth argument specifies the device (defaults to `CPU`).

## Security notes

- The AES-128-CBC encryption used here is a simplified example. Production deployments should use authenticated encryption (e.g., AES-256-GCM) and a proper key management solution.
- This approach protects models at rest on disk. Without hardware memory protection (e.g., Intel SGX, Intel TDX, or Intel TME), the decrypted model is still present in process memory at runtime.

## References

- [OpenVINO documentation: OpenVINO Security](https://docs.openvino.ai/2026/documentation/openvino-security.html)
- [OpenVINO blog: Use Encrypted Model with OpenVINO](https://blog.openvino.ai/blog-posts/use-encrypted-model-with-openvino)
- [OpenVINO GenAI encrypted model sample](https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/python/text_generation/encrypted_model_causal_lm.py)
