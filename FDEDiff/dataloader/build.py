import torch
from Models.ddpm.utils import fft_split
from torch.utils.data import DataLoader

from dataloader.basic import BasicPreprocess

def build_dataloader(config):
    cfg = config['dataloader']
    cfg_dataset = cfg['dataset']
    batch_size = cfg['batch_size']
    dataset = BasicPreprocess(
        cfg_dataset['name'],
        cfg_dataset['data_root'],
        cfg_dataset['keep_first'],
        cfg_dataset['window'],
        cfg_dataset['stride'],
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    return dataloader, dataset

def build_dataloader_LF(config, threshold):
    cfg = config['dataloader']
    cfg_dataset = cfg['dataset']
    batch_size = cfg['batch_size']
    dataset = BasicPreprocess(
        cfg_dataset['name'],
        cfg_dataset['data_root'],
        cfg_dataset['keep_first'],
        cfg_dataset['window'],
        cfg_dataset['stride'],
    )
    for idx in range(dataset.arrs.shape[0]):
        LF, HF = fft_split(torch.tensor(dataset.train_arrs[idx]), threshold)
        dataset.train_arrs[idx] = LF.numpy()
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    return dataloader, dataset    

def build_test_dataloader(config):
    cfg = config['dataloader']
    cfg_dataset = cfg['dataset']
    batch_size = cfg['batch_size']
    dataset = BasicPreprocess(
        f"test_{cfg_dataset['name']}",
        cfg_dataset['test_data'],
        cfg_dataset['keep_first'],
        cfg_dataset['window'],
        cfg_dataset['stride'],
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )
    return dataloader, dataset    