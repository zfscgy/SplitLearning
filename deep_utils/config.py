import torch

class GlobalConfig:
    device = "cuda"

    @staticmethod
    def set_device(device_name: str):
        GlobalConfig.device = device_name
        # torch.cuda.set_device(device_name)


