
import os
import numpy as np
import torch
import argparse
from runner import *


models = ['lin']
datasets = ['health']
N_explain = 10

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

run_benchmark(models, datasets, args.seed, N_explain)
