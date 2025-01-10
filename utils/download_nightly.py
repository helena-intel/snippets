#!/usr/bin/env python
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import requests


def find_nested_item(tree, path):
    current = tree
    for name in path:
        current = next(item for item in current.get("children", []) if item["name"] == name)
    return current


if sys.platform == "linux":
    ubuntu_version = subprocess.check_output(["lsb_release", "-rs"], text=True).split(".")[0]
    genai_package_str = f"ubuntu{ubuntu_version}"
    extension = ".tar.gz"
elif sys.platform == "win32":
    genai_package_str = "windows"
    extension = ".zip"

filetree = requests.get("https://storage.openvinotoolkit.org/filetree.json").json()

path = ["repositories", "openvino_genai", "packages", "nightly"]
nightly = find_nested_item({"children": filetree["children"]}, path)
latest_name = sorted([item["name"] for item in nightly["children"] if "latest" not in item["name"]], reverse=True)[0]
latest_files = next(item for item in nightly["children"] if item["name"] == latest_name)
latest_package = next(item["name"] for item in latest_files["children"] if re.match(f"openvino_genai_{genai_package_str}.*{extension}$", item["name"]))

package_url = f"https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/{latest_name}/{latest_package}"
response = requests.get(package_url)

destination_dir = Path("~/tools/openvino_genai").expanduser()

with tempfile.TemporaryDirectory() as tmpdir:
    archive_file = Path(tmpdir) / latest_package
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
