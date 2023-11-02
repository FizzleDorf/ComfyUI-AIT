import torch
import folder_paths
import comfy.utils
import os

if "ait" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ait"][0].append(os.path.join(folder_paths.models_dir, "ait"))
    folder_paths.folder_names_and_paths["ait"][1].add(".so")
    folder_paths.folder_names_and_paths["ait"][1].add(".dll")
else:
    folder_paths.folder_names_and_paths["ait"] = ([os.path.join(folder_paths.models_dir, "ait")], {".so", ".dll"})

def map_unet_params(pt_params):
    dim = 320
    params_ait = {}
    for key, arr in pt_params.items():
        if len(arr.shape) == 4:
            arr = arr.permute((0, 2, 3, 1)).contiguous()
        elif key.endswith("ff.net.0.proj.weight"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        elif key.endswith("ff.net.0.proj.bias"):
            w1, w2 = arr.chunk(2, dim=0)
            params_ait[key.replace(".", "_")] = w1
            params_ait[key.replace(".", "_").replace("proj", "gate")] = w2
            continue
        params_ait[key.replace(".", "_")] = arr.half()
    params_ait["arange"] = (
        torch.arange(start=0, end=dim // 2, dtype=torch.float32).cuda().half()
    )
    return params_ait

from aitemplate.compiler import Model

class AITModel:
    def __init__(self, ait_path):
        self.exe_module = Model(ait_path)

    def set_weights(self, sd):
        constants = map_unet_params(sd)
        self.exe_module.set_many_constants_with_tensors(constants)

    def apply_model(self, xc, t, context, y=None, control=None, transformer_options=None):
        xc = xc.permute((0, 2, 3, 1)).half().contiguous()
        output = [torch.empty_like(xc)]
        inputs = {"x": xc, "timesteps": t.half(), "context": context.half()}
        if y is not None:
            inputs['y'] = y.half()
        self.exe_module.run_with_tensors(inputs, output, graph_mode=False)
        return output[0].permute((0, 3, 1, 2))

    def unload_model(self):
        self.exe_module.close()


class AITPatch:
    def __init__(self, model, ait_model_path):
        self.model = model
        self.ait_model_path = ait_model_path
        self.ait_model = None

    def __call__(self, model_function, params):
        if self.ait_model is None:
            self.ait_model = AITModel(self.ait_model_path)
            sd = self.model.model_state_dict("diffusion_model.")
            sd = comfy.utils.state_dict_prefix_replace(sd, {"diffusion_model.": ""})
            self.ait_model.set_weights(sd)

        c_concat = params["c"].get("c_concat", None)

        # The BaseModel instance
        inner_model = self.model.model
        x = params["input"]
        sigma = params["timestep"]
        xc = inner_model.model_sampling.calculate_input(sigma, x)
        t = inner_model.model_sampling.timestep(sigma).float()
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)
        context = params["c"].get("c_crossattn")
        y = params["c"].get("y")
        control = params["c"].get("control")
        transformer_options = params["c"].get("transformer_options")
        out = self.ait_model.apply_model(xc, t, context, y, control, transformer_options)
        return inner_model.model_sampling.calculate_denoised(sigma, out, x)

    def to(self, a):
        if self.ait_model is not None:
            if a == torch.device("cpu"):
                self.ait_model.inner_model.unload_module()
                self.ait_model = None
                print("unloaded AIT")

class AIT_Unet_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL", ),
                              "ait_name": (folder_paths.get_filename_list("ait"), ),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_ait"

    CATEGORY = "loaders/AIT"

    def load_ait(self, model, ait_name):
        ait_path = folder_paths.get_full_path("ait", ait_name)
        patch = AITPatch(model, ait_path)
        model_ait = model.clone()
        model_ait.set_model_unet_function_wrapper(patch)
        return (model_ait,)

class AIT_VAE_Encode_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pixels": ("IMAGE", ),
                              "vae": ("VAE",),
                              "ait_name": (folder_paths.get_filename_list("ait"), ),
                             }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load_ait"

    CATEGORY = "loaders/AIT"

    @staticmethod
    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def load_ait(self, pixels, ait_name, vae):
        resolution = max(pixels.shape[1], pixels.shape[2])
        model_type = "vae_encode"

        # Clear any previously loaded VAE models
        if len(AITemplate.vae.keys()) > 0:
            to_delete = list(AITemplate.vae.keys())
            for key in to_delete:
                del AITemplate.vae[key]

        # Load the VAE module using the provided "ait_name"
        module_filename = folder_paths.get_full_path("ait", ait_name)
        if module_filename not in AITemplate.vae:
            AITemplate.vae[module_filename] = AITemplate.loader.load_module(module_filename)

        AITemplate.vae[module_filename] = AITemplate.loader.apply_vae(
            aitemplate_module=AITemplate.vae[module_filename],
            vae=AITemplate.loader.compvis_vae(vae.first_stage_model.state_dict()),
            encoder=True,
        )

        # Perform any required image processing here
        pixels = self.vae_encode_crop_pixels(pixels)
        pixels = pixels[:, :, :, :3]
        pixels = pixels.movedim(-1, 1)
        pixels = 2. * pixels - 1.

        samples = vae_inference(AITemplate.vae[module_filename], pixels, encoder=True)
        samples = samples.cpu()

        # Unload the module after inference
        del AITemplate.vae[module_filename]
        torch.cuda.empty_cache()

        return ({"samples": samples},)
