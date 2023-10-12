# ComfyUI-AIT
A ComfyUI implementation of Facebook Meta's [AITemplate](https://github.com/facebookincubator/AITemplate) repo for faster inference using cpp/cuda. The old AIT repo is still available for reference. You can find it [here](https://github.com/FizzleDorf/AIT).

## Installation
To install, you first need to apply the patch located in the patch folder. Here are the steps to do so: 
  - Make sure you are on the latest version of ComfyUI (`git pull` the root folder).
  - paste the patch (named `0001-AIT-Compile-support.patch`) into the ComfyUI root folder.
  - enter `git patch 0001-AIT-Compile-support.patch` into git.

## Modules
You can find already compiled modules [here](https://huggingface.co/Fizzledorf/AITemplateXL) otherwise you can compile your own modules (requires sm80 or higher NVIDIA GPU).
Place your downloaded or compiles Modules in the ComfyUI/models/ait directory.
While in ComfyUI, simply hook up the AIT_Unet_Loader to your workflow and select a module that is equal to or larger than the resolution you are generating at.

## Compiling
As stated earlier, you must have a sm80 or higher NVIDIA GP to compile modules. 

