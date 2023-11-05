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

    def apply_model(self, xc, t, c_crossattn, y=None, control=None, transformer_options=None, **kwargs):
        xc = xc.permute((0, 2, 3, 1)).half().contiguous()
        output = [torch.empty_like(xc)]
        inputs = {"x": xc, "timesteps": t.half(), "context": c_crossattn.half()}
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
        sigma = params["timestep"]
        t = self.model.model.model_sampling.timestep(sigma).float()
        x = self.model.model.model_sampling.calculate_input(sigma, params["input"])
        if c_concat is not None:
            xc = torch.cat([x] + [c_concat], dim=1)
        else:
            xc = x
        model_output = self.ait_model.apply_model(x, t, **params["c"]).float()
        return self.model.model.model_sampling.calculate_denoised(sigma, model_output, params["input"])

    def to(self, a):
        if self.ait_model is not None:
            if a == torch.device("cpu"):
                self.ait_model.unload_model()
                self.ait_model = None
                print("unloaded AIT")
        return self

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
        print(patch)
        return (model_ait,)



NODE_CLASS_MAPPINGS = {
  "AIT_Unet_Loader": AIT_Unet_Loader,
}
