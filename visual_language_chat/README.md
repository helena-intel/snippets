This is the visual language chat demo from OpenVINO GenAI: https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/visual_language_chat

No changes were made to the sample, only to CMakeLists.txt

## Build the sample

To build this sample, open a new Developer Command Prompt in the visual_language_chat directory from this repository and run:

> [!NOTE]
> Open a new Developer Command Prompt (for example in Windows Terminal) to make sure that you are not in an environment where you
> loaded OpenVINO already, for example with Python, or from a setupvars from a different version.

```sh
source \path\to\genai_archive\setupvars.bat
mkdir build
cd build
cmake ..
cmake --build . --config Release
cmake --install . --prefix %USERPROFILE%\visual_language_chat_build
```

The last step is optional, after the `--build` step your sample can be run from the `build` directory with `Release\visual_language_chat.exe <MODEL_DIR> <IMAGE_FILE_OR_DIR>`.
The benefit of the last step is that it installs all required DLLs in the same directory as the executable, so it can be ran without having to run setupvars.bat first

## Run the sample

To run this, download an OpenVINO VLM from the Hugging Face hub. Run the following in a python environment with `huggingface-hub` installed (this is included in any transformers or optimum environment):

```
huggingface-cli download helenai/InternVL2-2B-ov --local-dir InternVL2-2B-ov
```

and a test image:

```
curl -O "https://storage.openvinotoolkit.org/test_data/images/dog.jpg"
```

Then deactivate any Python environments, or open a new command prompt, and run

```
visual_language_chat.exe InternVL2-2B-ov dog.jpg
```
