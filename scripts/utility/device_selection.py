import torch

def send_to_preferred_device(obj):
    if obj.device != torch.device("vulkan") and torch.is_vulkan_available():
        print("[INFO]: sending to Vulkan device")
        obj = obj.to(torch.device("vulkan"))
    elif obj.device != torch.device("cuda") and torch.cuda.is_available():
        print("[INFO]: sending to CUDA device")
        obj = obj.to(torch.device("cuda"))
    else:
        print("[INFO]: skipped sending to device")
    return obj

def get_preferred_device():
    if torch.is_vulkan_available():
        print("[INFO]: using Vulkan")
        return "vulkan"
    elif torch.cuda.is_available():
        print("[INFO]: using CUDA")
        return "cuda"
    else:
        print("[INFO]: using CPU")
        return "cpu"
