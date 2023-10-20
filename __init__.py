from .ait_load import AIT_Unet_Loader#, AIT_VAE_Encode_Loader

NODE_CLASS_MAPPINGS = {
    "AIT_Unet_Loader":AIT_Unet_Loader,
    #"AIT_VAE_Encode_Loader":AIT_VAE_Encode_Loader,
}

print('\033[34mAIT Nodes: \033[92mLoaded\033[0m')