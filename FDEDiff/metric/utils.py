import numpy as np
import scipy
import torch
import yaml

import os
import random


def display_scores(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., len(results) - 1)
   # 95% confidence interval
   mean_rounded = round(mean, 3)
    
   sigma_decimal = str(round(sigma, 3))[1:]
   print('Final Score: ', f'{mean} +- {sigma}')
   return f'Final Score: {mean} +- {sigma} | {mean_rounded} +- {sigma_decimal}'


def load_npz(dataset_dir, origin_path='origin.npz', gen_path='gen.npz'):
    ori_data = np.load(os.path.join(dataset_dir, origin_path))['data']
    gen_data = np.load(os.path.join(dataset_dir, gen_path))['data']
    return ori_data, gen_data


def load_npy(dataset_dir, origin_path='origin.npy', gen_path='gen.npy'):
    ori_data = np.load(os.path.join(dataset_dir, origin_path))
    gen_data = np.load(os.path.join(dataset_dir, gen_path))
    return ori_data, gen_data


def setseed(seed = 9163):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config