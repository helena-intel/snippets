"""
Load an encrypted OpenVINO IR model, compile it, and optionally run inference.

Demonstrates decrypting a model entirely in memory (no decrypted files on disk).
If the model has static shapes and floating point inputs, dummy inference is also run.

Requirements:
    pip install openvino cryptography

Usage:
    python infer_encrypted.py model_enc.xml --key <hex> --iv <hex>
"""

import argparse
from pathlib import Path

import numpy as np
import openvino as ov
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7


def decrypt(path, key_hex, iv_hex):
    key = bytes.fromhex(key_hex)
    iv = bytes.fromhex(iv_hex)
    cipher = Cipher(algorithms.AES128(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(path.read_bytes()) + decryptor.finalize()
    unpadder = PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


parser = argparse.ArgumentParser(description="Encrypted model inference demo")
parser.add_argument("xml", type=Path, help="Encrypted .xml file")
parser.add_argument("--key", required=True, help="AES-128 key in hex")
parser.add_argument("--iv", required=True, help="AES-128 IV in hex")
parser.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU)")
args = parser.parse_args()

bin_path = args.xml.with_suffix(".bin")
xml_bytes = decrypt(args.xml, args.key, args.iv)
bin_bytes = decrypt(bin_path, args.key, args.iv)

core = ov.Core()
model = core.read_model(model=xml_bytes, weights=bin_bytes)
compiled = core.compile_model(model, args.device)

print("Model compiled successfully from encrypted files.")
print("\nInputs:")
for inp in compiled.inputs:
    print(f"  {inp.get_any_name()}: {inp.get_partial_shape()}")
print("Outputs:")
for out in compiled.outputs:
    print(f"  {out.get_any_name()}: {out.get_partial_shape()}")

try:
    inputs = {}
    for inp in compiled.inputs:
        shape = inp.get_partial_shape()
        if shape.is_dynamic:
            raise ValueError(f"Input '{inp.get_any_name()}' has dynamic shape {shape}")
        static_shape = [d.get_length() for d in shape]
        inputs[inp.get_any_name()] = np.random.rand(*static_shape).astype(np.float32)

    result = compiled(inputs)
    print("\nInference result:")
    for out in compiled.outputs:
        print(f"  {out.get_any_name()}: shape={result[out].shape}")
except Exception as e:
    print(f"\nSkipping inference: {e}")
