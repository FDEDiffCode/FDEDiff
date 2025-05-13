import os
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class BasicPreprocess(Dataset):
    def __init__(
        self,
        name,
        data_root,
        keep_first = True, # 第一列如果是时间就设置为False
        window = 24,
        stride = 1,
    ):
        super(BasicPreprocess, self).__init__()
        self.name = name
        self.rawdata, self.scaler = self.read_data(data_root, keep_first)
        self.window = window
        self.stride = stride
        self.len, self.var_num = self.rawdata.shape
        self.data = self.normalize2D(self.rawdata)
        self.arrs = self.slide_window(self.data)
        self.N = self.arrs.shape[0]
        self.train_arrs = self.normalize_train(self.arrs)

    def save_npy(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'origin_{self.name}_{self.window}L_{self.stride}S_norm.npy'), self.arrs)
        np.save(os.path.join(save_dir, f'origin_{self.name}_{self.window}L_{self.stride}S_raw.npy'), self.inverse(self.arrs))

    def save_gen(self, samples, save_dir):
        samples = self.inverse_train(samples)
        np.save(os.path.join(save_dir, f'fake_{self.name}_{self.window}L_{self.stride}S_norm.npy'), samples)

    @staticmethod
    def read_data(filepath, keep_first = True):
        df = pd.read_csv(filepath, header=0)
        if not keep_first:
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def normalize(self, sq): # (N, window, var_num)
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        return d.reshape(-1, self.window, self.var_num)
    
    def inverse(self, sq): # (N, window, var_num)
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.inverse_transform(d)
        return d.reshape(-1, self.window, self.var_num)

    def normalize2D(self, rawdata): # (len, var_num)
        data = self.scaler.transform(rawdata)
        return data
    
    def inverse2D(self, data): # (len, var_num)
        return self.scaler.inverse_transform(data)

    @staticmethod
    def normalize_train(rawdata): # [0, 1]
        return rawdata * 2 - 1 # [-1, 1]
    
    @staticmethod
    def inverse_train(data): # [-1, 1]
        return (data + 1) * 0.5 # [0, 1]

    def slide_window(self, data):
        x = []
        for i in range(0, self.len, self.stride):
            start = i
            end = i + self.window
            if end > self.len:
                break
            x.append(data[start: end, :])
        x = np.asarray(x)
        return x
    
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return torch.from_numpy(self.train_arrs[idx, :, :]).float()