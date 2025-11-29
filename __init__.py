import torch
import logging
import os
import copy
from pathlib import Path
import folder_paths
import comfy.model_management as mm
from nodes import NODE_CLASS_MAPPINGS as GLOBAL_NODE_CLASS_MAPPINGS
from .device_utils import get_device_list, is_accelerator_available

# --- DisTorch V2 Logging Configuration ---
# Set to "E" for Engineering (DEBUG) or "P" for Production (INFO)
LOG_LEVEL = "P"

# Configure logger
logger = logging.getLogger("MultiGPU")
logger.propagate = False

if not logger.handlers:
    log_level = logging.DEBUG if LOG_LEVEL == "E" else logging.INFO
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.info(f"[MultiGPU Initialization] Logger initialized with level: {logging.getLevelName(log_level)}")


# Global device state management
current_device = mm.get_torch_device()
current_text_encoder_device = mm.text_encoder_device()

def set_current_device(device):
    global current_device
    current_device = device
    logger.info(f"[MultiGPU Initialization] current_device set to: {device}")

def set_current_text_encoder_device(device):
    global current_text_encoder_device
    current_text_encoder_device = device
    logger.info(f"[MultiGPU Initialization] current_text_encoder_device set to: {device}")

def override_class(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):

            if device is not None:
                set_current_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)

            return out

    return NodeOverride

def override_class_clip(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            if device is not None:
                set_current_text_encoder_device(device)
            kwargs['device'] = 'default'
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            return out

    return NodeOverride

def override_class_clip_no_device(cls):
    class NodeOverride(cls):
        @classmethod
        def INPUT_TYPES(s):
            inputs = copy.deepcopy(cls.INPUT_TYPES())
            devices = get_device_list()
            default_device = devices[1] if len(devices) > 1 else devices[0]
            inputs["optional"] = inputs.get("optional", {})
            inputs["optional"]["device"] = (devices, {"default": default_device})
            return inputs

        CATEGORY = "multigpu"
        FUNCTION = "override"

        def override(self, *args, device=None, **kwargs):
            if device is not None:
                set_current_text_encoder_device(device)
            fn = getattr(super(), cls.FUNCTION)
            out = fn(*args, **kwargs)
            
            return out

    return NodeOverride


def get_torch_device_patched():
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_device) if str(current_device) in devs else torch.device("cpu")
    logger.debug(f"[MultiGPU Core Patching] get_torch_device_patched returning device: {device} (current_device={current_device})")
    return device

def text_encoder_device_patched():
    device = None
    if (not is_accelerator_available() or mm.cpu_state == mm.CPUState.CPU or "cpu" in str(current_text_encoder_device).lower()):
        device = torch.device("cpu")
    else:
        devs = set(get_device_list())
        device = torch.device(current_text_encoder_device) if str(current_text_encoder_device) in devs else torch.device("cpu")
    logger.debug(f"[MultiGPU Core Patching] text_encoder_device_patched returning device: {device} (current_text_encoder_device={current_text_encoder_device})")
    return device


logger.info(f"[MultiGPU Core Patching] Patching mm.get_torch_device, mm.text_encoder_device, and mm.text_encoder_initial_device")
logger.debug(f"[MultiGPU DEBUG] Initial current_device: {current_device}")
logger.debug(f"[MultiGPU DEBUG] Initial current_text_encoder_device: {current_text_encoder_device}")
mm.get_torch_device = get_torch_device_patched
mm.text_encoder_device = text_encoder_device_patched

def check_module_exists(module_path):
    full_path = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0], module_path)
    logger.debug(f"[MultiGPU] Checking for module at {full_path}")
    if not os.path.exists(full_path):
        logger.debug(f"[MultiGPU] Module {module_path} not found - skipping")
        return False
    logger.debug(f"[MultiGPU] Found {module_path}, creating compatible MultiGPU nodes")
    return True

# Import from nodes.py
from .nodes import (
    FossielDeviceSelectorMultiGPU,
    FossielHunyuanVideoEmbeddingsAdapter,
    FossielUnetLoaderGGUF,
    FossielUnetLoaderGGUFAdvanced,
    FossielCLIPLoaderGGUF,
    FossielDualCLIPLoaderGGUF,
    FossielTripleCLIPLoaderGGUF,
    FossielQuadrupleCLIPLoaderGGUF,
    FossielLTXVLoader,
    FossielFlorence2ModelLoader,
    FossielDownloadAndLoadFlorence2Model,
    FossielCheckpointLoaderNF4,
    FossielLoadFluxControlNet,
    FossielMMAudioModelLoader,
    FossielMMAudioFeatureUtilsLoader,
    FossielMMAudioSampler,
    FossielPulidModelLoader,
    FossielPulidInsightFaceLoader,
    FossielPulidEvaClipLoader,
    FossielHyVideoModelLoader,
    FossielHyVideoVAELoader,
    FossielDownloadAndLoadHyVideoTextEncoder,
)

# Import from wanvideo.py
from .wanvideo import (
    FossielWanVideoModelLoader,
    FossielWanVideoModelLoader_2,
    FossielWanVideoVAELoader,
    FossielLoadWanVideoT5TextEncoder,
    FossielLoadWanVideoClipTextEncoder,
    FossielWanVideoTextEncode,
    FossielWanVideoBlockSwap,
    FossielWanVideoSampler
)

# Import from distorch.py
from .distorch import (
    model_allocation_store,
    create_model_hash,
    register_patched_ggufmodelpatcher,
    analyze_ggml_loading,
    calculate_vvram_allocation_string,
    override_class_with_distorch_gguf,
    override_class_with_distorch_gguf_v2,
    override_class_with_distorch_clip,
    override_class_with_distorch_clip_no_device,
    override_class_with_distorch
)

# Import from distorch_2.py for DisTorch v2 SafeTensor support
from .distorch_2 import (
    safetensor_allocation_store,
    create_safetensor_model_hash,
    register_patched_safetensor_modelpatcher,
    analyze_safetensor_loading,
    calculate_safetensor_vvram_allocation,
    override_class_with_distorch_safetensor_v2,
    override_class_with_distorch_safetensor_v2_clip,
    override_class_with_distorch_safetensor_v2_clip_no_device
)

# Import advanced checkpoint loaders
from .checkpoint_multigpu import (
    FossielCheckpointLoaderAdvancedMultiGPU,
    FossielCheckpointLoaderAdvancedDisTorch2MultiGPU
)

# Initialize NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "FossielDeviceSelectorMultiGPU": FossielDeviceSelectorMultiGPU,
    "FossielHunyuanVideoEmbeddingsAdapter": FossielHunyuanVideoEmbeddingsAdapter,
    "FossielCheckpointLoaderAdvancedMultiGPU": FossielCheckpointLoaderAdvancedMultiGPU,
    "FossielCheckpointLoaderAdvancedDisTorch2MultiGPU": FossielCheckpointLoaderAdvancedDisTorch2MultiGPU,
}

# Standard MultiGPU nodes
NODE_CLASS_MAPPINGS["FossielUNETLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["FossielVAELoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["FossielCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["FossielDualCLIPLoaderMultiGPU"] = override_class_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielTripleCLIPLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
if "QuadrupleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielQuadrupleCLIPLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["FossielCLIPVisionLoaderMultiGPU"] = override_class_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["FossielCheckpointLoaderSimpleMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["FossielControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
if "DiffusersLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielDiffusersLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
if "DiffControlNetLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielDiffControlNetLoaderMultiGPU"] = override_class(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

# DisTorch 2 SafeTensor nodes for FLUX and other safetensor models
NODE_CLASS_MAPPINGS["FossielUNETLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["UNETLoader"])
NODE_CLASS_MAPPINGS["FossielVAELoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["VAELoader"])
NODE_CLASS_MAPPINGS["FossielCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip(GLOBAL_NODE_CLASS_MAPPINGS["CLIPLoader"])
NODE_CLASS_MAPPINGS["FossielDualCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip(GLOBAL_NODE_CLASS_MAPPINGS["DualCLIPLoader"])
if "TripleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielTripleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["TripleCLIPLoader"])
if "QuadrupleCLIPLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielQuadrupleCLIPLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["QuadrupleCLIPLoader"])
NODE_CLASS_MAPPINGS["FossielCLIPVisionLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2_clip_no_device(GLOBAL_NODE_CLASS_MAPPINGS["CLIPVisionLoader"])
NODE_CLASS_MAPPINGS["FossielCheckpointLoaderSimpleDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"])
NODE_CLASS_MAPPINGS["FossielControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["ControlNetLoader"])
if "DiffusersLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielDiffusersLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffusersLoader"])
if "DiffControlNetLoader" in GLOBAL_NODE_CLASS_MAPPINGS:
    NODE_CLASS_MAPPINGS["FossielDiffControlNetLoaderDisTorch2MultiGPU"] = override_class_with_distorch_safetensor_v2(GLOBAL_NODE_CLASS_MAPPINGS["DiffControlNetLoader"])

# --- Registration Table ---
logger.info("[MultiGPU] Initiating custom_node Registration. . .")
dash_line = "-" * 47
fmt_reg = "{:<30}{:>5}{:>10}"
logger.info(dash_line)
logger.info(fmt_reg.format("custom_node", "Found", "Nodes"))
logger.info(dash_line)

registration_data = []

def register_and_count(module_names, node_map):
    found = False
    for name in module_names:
        if check_module_exists(name):
            found = True
            break
    
    count = 0
    if found:
        initial_len = len(NODE_CLASS_MAPPINGS)
        for key, value in node_map.items():
            NODE_CLASS_MAPPINGS[key] = value
        count = len(NODE_CLASS_MAPPINGS) - initial_len
        
    registration_data.append({"name": module_names[0], "found": "Y" if found else "N", "count": count})
    return found

# ComfyUI-LTXVideo
ltx_nodes = {"FossielLTXVLoaderMultiGPU": override_class(FossielLTXVLoader)}
register_and_count(["ComfyUI-LTXVideo", "comfyui-ltxvideo"], ltx_nodes)

# ComfyUI-Florence2
florence_nodes = {
    "FossielFlorence2ModelLoaderMultiGPU": override_class(FossielFlorence2ModelLoader),
    "FossielDownloadAndLoadFlorence2ModelMultiGPU": override_class(FossielDownloadAndLoadFlorence2Model)
}
register_and_count(["ComfyUI-Florence2", "comfyui-florence2"], florence_nodes)

# ComfyUI_bitsandbytes_NF4
nf4_nodes = {"FossielCheckpointLoaderNF4MultiGPU": override_class(FossielCheckpointLoaderNF4)}
register_and_count(["ComfyUI_bitsandbytes_NF4", "comfyui_bitsandbytes_nf4"], nf4_nodes)

# x-flux-comfyui
flux_controlnet_nodes = {"FossielLoadFluxControlNetMultiGPU": override_class(FossielLoadFluxControlNet)}
register_and_count(["x-flux-comfyui"], flux_controlnet_nodes)

# ComfyUI-MMAudio
mmaudio_nodes = {
    "FossielMMAudioModelLoaderMultiGPU": override_class(FossielMMAudioModelLoader),
    "FossielMMAudioFeatureUtilsLoaderMultiGPU": override_class(FossielMMAudioFeatureUtilsLoader),
    "FossielMMAudioSamplerMultiGPU": override_class(FossielMMAudioSampler)
}
register_and_count(["ComfyUI-MMAudio", "comfyui-mmaudio"], mmaudio_nodes)

# ComfyUI-GGUF
gguf_nodes = {
    "FossielUnetLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_gguf(FossielUnetLoaderGGUF),
    "FossielUnetLoaderGGUFAdvancedDisTorchMultiGPU": override_class_with_distorch_gguf(FossielUnetLoaderGGUFAdvanced),
    "FossielCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip(FossielCLIPLoaderGGUF),
    "FossielDualCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip(FossielDualCLIPLoaderGGUF),
    "FossielTripleCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip_no_device(FossielTripleCLIPLoaderGGUF),
    "FossielQuadrupleCLIPLoaderGGUFDisTorchMultiGPU": override_class_with_distorch_clip_no_device(FossielQuadrupleCLIPLoaderGGUF),
    "FossielUnetLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2(FossielUnetLoaderGGUF),
    "FossielUnetLoaderGGUFAdvancedDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2(FossielUnetLoaderGGUFAdvanced),
    "FossielCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip(FossielCLIPLoaderGGUF),
    "FossielDualCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip(FossielDualCLIPLoaderGGUF),
    "FossielTripleCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip_no_device(FossielTripleCLIPLoaderGGUF),
    "FossielQuadrupleCLIPLoaderGGUFDisTorch2MultiGPU": override_class_with_distorch_safetensor_v2_clip_no_device(FossielQuadrupleCLIPLoaderGGUF),
    "FossielUnetLoaderGGUFMultiGPU": override_class(FossielUnetLoaderGGUF),
    "FossielUnetLoaderGGUFAdvancedMultiGPU": override_class(FossielUnetLoaderGGUFAdvanced),
    "FossielCLIPLoaderGGUFMultiGPU": override_class_clip(FossielCLIPLoaderGGUF),
    "FossielDualCLIPLoaderGGUFMultiGPU": override_class_clip(FossielDualCLIPLoaderGGUF),
    "FossielTripleCLIPLoaderGGUFMultiGPU": override_class_clip_no_device(FossielTripleCLIPLoaderGGUF),
    "FossielQuadrupleCLIPLoaderGGUFMultiGPU": override_class_clip_no_device(FossielQuadrupleCLIPLoaderGGUF)
}
register_and_count(["ComfyUI-GGUF", "comfyui-gguf"], gguf_nodes)

# PuLID_ComfyUI
pulid_nodes = {
    "FossielPulidModelLoaderMultiGPU": override_class(FossielPulidModelLoader),
    "FossielPulidInsightFaceLoaderMultiGPU": override_class(FossielPulidInsightFaceLoader),
    "FossielPulidEvaClipLoaderMultiGPU": override_class(FossielPulidEvaClipLoader)
}
register_and_count(["PuLID_ComfyUI", "pulid_comfyui"], pulid_nodes)

# ComfyUI-HunyuanVideoWrapper
hunyuan_nodes = {
    "FossielHyVideoModelLoaderMultiGPU": override_class(FossielHyVideoModelLoader),
    "FossielHyVideoVAELoaderMultiGPU": override_class(FossielHyVideoVAELoader),
    "FossielDownloadAndLoadHyVideoTextEncoderMultiGPU": override_class(FossielDownloadAndLoadHyVideoTextEncoder)
}
register_and_count(["ComfyUI-HunyuanVideoWrapper", "comfyui-hunyuanvideowrapper"], hunyuan_nodes)

# ComfyUI-WanVideoWrapper
wanvideo_nodes = {
    "FossielWanVideoModelLoaderMultiGPU": FossielWanVideoModelLoader,
    "FossielWanVideoModelLoaderMultiGPU_2": FossielWanVideoModelLoader_2,
    "FossielWanVideoVAELoaderMultiGPU": FossielWanVideoVAELoader,
    "FossielLoadWanVideoT5TextEncoderMultiGPU": FossielLoadWanVideoT5TextEncoder,
    "FossielLoadWanVideoClipTextEncoderMultiGPU": FossielLoadWanVideoClipTextEncoder,
    "FossielWanVideoTextEncodeMultiGPU": FossielWanVideoTextEncode,
    "FossielWanVideoBlockSwapMultiGPU": FossielWanVideoBlockSwap,
    "FossielWanVideoSamplerMultiGPU": FossielWanVideoSampler
}
register_and_count(["ComfyUI-WanVideoWrapper", "comfyui-wanvideowrapper"], wanvideo_nodes)

# Print the registration table
for item in registration_data:
    logger.info(fmt_reg.format(item['name'], item['found'], str(item['count'])))
logger.info(dash_line)


logger.info(f"[MultiGPU] Registration complete. Final mappings: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
