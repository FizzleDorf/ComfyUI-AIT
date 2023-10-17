# ComfyUI-AIT

A ComfyUI implementation of Facebook Meta's [AITemplate](https://github.com/facebookincubator/AITemplate) repo for faster inference using cpp/cuda. The old AIT repo is still available for reference. You can find it [here](https://github.com/FizzleDorf/AIT). The node in the UI is located under loaders->AIT. This new repo is behiond but is a much more stable foundation to keep AIT online. *Please be patient as the repo will eventually include the same features as before.*

## Installation

To install the repo, simply git bash inside the custom nodes directory and use the command: 
`git clone https://github.com/FizzleDorf/ComfyUI-AIT.git`

You need to apply the patch located in the patch folder. Here are the steps to do so: 
  - Make sure you are on the latest version of ComfyUI (`git pull` the root folder).
  - paste the patch (named `0001-AIT-Compile-support.patch`) into the ComfyUI root folder.
  - enter `git apply 0001-AIT-Compile-support.patch` into git bash.
*Note:this is temporarily in place until this is included in base comfy* 

## Modules

You can find already compiled modules [here](https://huggingface.co/Fizzledorf/AITemplateXL) otherwise you can compile your own modules (requires sm80 or higher NVIDIA GPU). *An updated wheel will be provided for Windows modules.*
Place your downloaded or compiles Modules in the ComfyUI/models/ait directory.
While in ComfyUI, simply hook up the AIT_Unet_Loader to your workflow and select a module that is equal to or larger than the resolution you are generating at.

