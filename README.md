# ComfyUI-AIT

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) implementation of Facebook Meta's [AITemplate](https://github.com/facebookincubator/AITemplate) repo for faster inference using cpp/cuda. The old AIT repo is still available for reference. You can find it [here](https://github.com/FizzleDorf/AIT). The node in the UI is located under loaders->AIT. This new repo is behiond but is a much more stable foundation to keep AIT online. *Please be patient as the repo will eventually include the same features as before.*

## Installation

This repo can be downloaded using [ComfyUI manager](https://github.com/ltdrdata/ComfyUI-Manager).

To install the repo manually, simply git bash inside the `custom_nodes` directory and use the command `git clone https://github.com/FizzleDorf/ComfyUI-AIT.git`

### Linux
*This is for compilation only, you can do the Linux install for inference only*
- open a terminal pathed to the current folder and use `git clone --recursive https://github.com/facebookincubator/AITemplate`.
- path to `cd python`.
- run `python setup.py bdist_wheel`.
- **If you are using a virtual environment, make sure it's activated before installing the wheel.**
- `pip install dist/*.whl --force-reinstall`
- copy and paste the `3rdparty` directory to the `Lib` directory of your venv or python install.

### Windows
*This is for compilation only, you can do the Linux install for inference only*
- Extract the `ait_windows.zip` file in `\ComfyUI-AIT\compile` directory.
- open a cmd pathed to the current folder and use `git clone --recursive ait_windows.bundle -b fixes`.
- path to `cd ait_windows/python`.
- run `python setup.py bdist_wheel`.
- **If you cloned your ComfyUI install and you are using a virtual environment, make sure it's activated before installing the wheel.**
- `pip install dist/*.whl --force-reinstall` for cloned comfy installs or `.\python_embed\python.exe -s -m pip install` for packaged comfy.
- copy and paste the `3rdparty` directory to the `Lib` directory of your venv or python install.

## Modules

You can find already [compiled modules here](https://huggingface.co/Fizzledorf/AITemplateXL) otherwise you can compile your own modules (requires sm80 or higher NVIDIA GPU). *An updated wheel will be provided for Windows modules.*
Place your downloaded or compiles Modules in the ComfyUI/models/ait directory.
While in ComfyUI, simply hook up the AIT_Unet_Loader to your workflow and select a module that is equal to or larger than the resolution you are generating at.
