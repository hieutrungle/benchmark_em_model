import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
import logger
import os
import json
import shutil

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
NUM_GPUS = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])

# DEVICE = torch.device(str(os.environ.get("DEVICE", "cpu")))
# NUM_GPUS = int(os.environ.get("NUM_GPUS", 0))


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def log_args_and_device_info(args):
    """Logs arguments to the console."""
    logger.log(f"{'*'*23} {'train'.upper()} BEGIN {'*'*23}")
    message = "\n"
    for k, v in args.__dict__.items():
        if isinstance(v, str):
            message += f"{k} = '{v}'\n"
        else:
            message += f"{k} = {v}\n"

    # Additional Info when using cuda
    if DEVICE.type == "cuda":
        message += f"\nUsing device: {str(DEVICE)}\n"
        for i in range(NUM_GPUS):
            mem_allot = round(torch.cuda.memory_allocated(i) / 1024**3, 1)
            mem_cached = round(torch.cuda.memory_reserved(i) / 1024**3, 1)
            message += f"{str(torch.cuda.get_device_name(i))}\n"
            message += "Memory Usage: " + "\n"
            message += "Allocated: " + str(mem_allot) + " GB" + "\n"
            message += "Cached: " + str(mem_cached) + " GB" + "\n"
    logger.log(f"{message}")
    logger.log(f"Pytorch version: {torch.__version__}\n")


class NpEncoder(json.JSONEncoder):
    # json format for saving numpy array
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def mkdir_storage(model_dir, resume={}):
    if os.path.exists(os.path.join(model_dir, "summaries")):
        if len(resume) == 0:
            # val = input("The model directory %s exists. Overwrite? (y/n) " % model_dir)
            # print()
            # if val == 'y':
            if os.path.exists(os.path.join(model_dir, "summaries")):
                shutil.rmtree(os.path.join(model_dir, "summaries"))
            if os.path.exists(os.path.join(model_dir, "checkpoints")):
                shutil.rmtree(os.path.join(model_dir, "checkpoints"))

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, "summaries")
    mkdir_if_not_exist(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    mkdir_if_not_exist(checkpoints_dir)
    return summaries_dir, checkpoints_dir


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
