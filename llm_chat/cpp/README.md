# OpenVINO C++ chat sample

## Prepare environment:

Download and uncompress an [OpenVINO archive](https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/).
setupvars.bat|ps1|sh are in the root of the uncompressed archive file.

### Windows

Run this in a Developer Command Prompt / PowerShell

```shell
\path\to\setupvars.bat|ps1
```

### Linux

```shell
source /path/to/setupvars.sh
```

## Build:

```shell
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Run 

> [!IMPORTANT]  
> Run/source setupvars before running the application

Usage:

```
llm_chat model_path device
```

### Windows example:

```
Release\llm_chat.exe \path\to\model GPU
```

### Linux example:

```
./llm_chat /path/to/model GPU
```

## Deploy

To deploy this application including necessary DLLs, see these [CMakeLists](https://github.com/helena-intel/snippets/tree/main/genai_cmakelists).
