#!/usr/bin/env python

"""
Download OpenVINO GenAI release archive and show setupvars command
The archive is extracted in $HOME/tools/openvino_genai/openvino_genai_version
Modify `destination_dir` to use a different directory
This is intended as a simple script for development use, not expected to be robust or for production use

Requirements: `pip install requests`

Usage: python download_release.py version
Example: python download_release.py 2025.2
"""

import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests

destination_dir = Path("~/tools/openvino_genai").expanduser()

version = sys.argv[1]

if sys.platform == "linux":
    ubuntu_version = subprocess.check_output(["lsb_release", "-rs"], text=True).split(".")[0]
    genai_package_str = f"ubuntu{ubuntu_version}"
    extension = ".tar.gz"
    platform = "linux"
elif sys.platform == "win32":
    genai_package_str = "windows"
    extension = ".zip"
    platform = "windows"

archive_filename = rf"openvino_genai_{genai_package_str}_{version}.0.0_x86_64{extension}"
package_url = rf"https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/{version}/{platform}/{archive_filename}"

response = requests.get(package_url)

with tempfile.TemporaryDirectory() as tmpdir:
    archive_file = Path(tmpdir) / archive_filename
    dirname = re.split(extension, archive_file.name)[0]
    with open(archive_file, "wb") as file:
        file.write(response.content)
        if extension == ".tar.gz":
            with tarfile.open(archive_file, "r:gz") as tar:
                tar.extractall(destination_dir)
            setupvars = f"source {destination_dir}/{dirname}/setupvars.sh"
        else:
            with zipfile.ZipFile(archive_file, "r") as zip:
                zip.extractall(destination_dir)
            setupvars = rf"{destination_dir}\{dirname}\setupvars.bat"

print(f"Downloaded {destination_dir / dirname}")
print(setupvars)
