import os
import json
import random
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed:int=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_tuple(s):
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except:
        raise argparse.ArgumentTypeError("Tuple must be in the format (x, y)")
    

def load_configs(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def print_f(contents, file):
    print(contents)
    print(contents, file=file)