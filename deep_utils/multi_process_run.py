import sys
import uuid

import torch
from typing import List, Callable, Union
# from multiprocessing import Process, Manager
from torch.multiprocessing import Process, Manager
# from pathos.multiprocessing import Process, Manager

from deep_utils.config import GlobalConfig


def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = uuid.uuid4().hex
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result


def with_device(devices, func, args: list, i: int, return_dict: dict):
    if devices[i].startswith("cuda"):
        with torch.cuda.device(devices[i]):
            print(f"Current process: {i}, device: {GlobalConfig.device}")
            return_dict[i] = func(*args)
    else:
        print(f"Current process: {i}, device: {GlobalConfig.device}")
        return_dict[i] = func(*args)

def parallel_process(funcs: Union[Callable, List[Callable]], args: List = None, devices: List[str] = None):
    manager = Manager()
    return_dict = manager.dict()

    if not isinstance(funcs, list):
        if args is not None:
            funcs = [funcs] * len(args)
        else:
            funcs = [funcs]

    if args is None:
        args = [[]] * len(funcs)

    if devices is None:
        devices = ['cuda:0' for _ in funcs]

    current_processes = []

    for i, _ in enumerate(funcs):
        current_processes.append(Process(target=with_device,
                                         args=(devices, funcs[i], args[i], i, return_dict)))
        current_processes[-1].start()

    for p in current_processes:
        p.join()

    return [return_dict[i] for i in range(len(funcs))]
