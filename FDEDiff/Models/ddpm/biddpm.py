import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
import einops
from Models.ddpm.utils import fft_split
class Biddpm(nn.Module):
    def __init__(self, diffusion_model_LF, diffusion_model_HF, split_threshold, augment_alpha = 1., augment_mode = 'split'):
        """
        初始化 Biddpm 模型
        :param diffusion_model_LF: 用于处理低频部分的 Diffusion 模型
        :param diffusion_model_HF: 用于处理高频部分及其叠加信息的 Diffusion 模型
        :param split_threshold: 用于区分低频和高频的阈值，控制低频和高频的分离
        :param timesteps: Diffusion 模型的时间步长
        """
        super(Biddpm, self).__init__()
        self.diffusion_model_LF = diffusion_model_LF  # 低频 Diffusion 模型
        self.diffusion_model_HF = diffusion_model_HF  # 高频 Diffusion 模型
        self.split_threshold = split_threshold
        self.augment_alpha = augment_alpha
        self.augment_mode = augment_mode # split: 分为高频和低频； appended： 高频增强后加在一起
    
    def forward(self, x):
        """
        模型的前向传播过程
        :param x: 输入的时间序列，形状为 (B, T, D)
        :return: 最终的时间序列预测，形状为 (B, T, D)
        """
        B, T, D = x.shape
        LF, HF = fft_split(x, self.split_threshold)
        # combined_LF_HF = hat_LF + HF  #  (B, T, D)
        # print(f'[pre_cond] LF_shape {LF.shape} HF_shape {HF.shape}')
        # LF = self.diffusion_model_LF.infer(LF)
        # print(f'[conded] LF_shape {LF.shape} HF_shape {HF.shape}')
        input = HF * self.augment_alpha
        if self.augment_mode != 'split':
        #    print('Check split √')
            input = input + LF
        output = self.diffusion_model_HF(input, LF)  #  (B, T, D)
        
        return output

    def infer(self, x, steps=1, noise_fn=torch.randn_like):
        B, T, D = x.shape
        LF, HF = fft_split(x, self.split_threshold)
        LF_hat = self.diffusion_model_LF.infer(LF)
        HF_hat = self.diffusion_model_HF.sample(HF.shape, LF_hat)
        if self.augment_alpha == 'split':
            x_hat = HF_hat / self.augment_alpha + LF_hat
        else :
            x_hat = ( HF_hat - LF_hat ) / self.augment_alpha + LF_hat
        # x_hat = self.fft_reconstruct(LF_hat, HF_hat) # 拼接后的代码
        return torch.clamp(x_hat, min=-1.0, max=1.0)
    def infer_reference(self, x, steps=1, noise_fn=torch.randn_like):
        B, T, D = x.shape
        LF, HF = fft_split(x, self.split_threshold)
        HF_hat = self.diffusion_model_HF.sample(HF.shape, LF)
        if self.augment_alpha == 'split':
            x_hat = HF_hat / self.augment_alpha + LF_hat
        else :
            x_hat = ( HF_hat - LF_hat ) / self.augment_alpha + LF_hat
        # x_hat = self.fft_reconstruct(LF, HF_hat) # 拼接后的代码
        return torch.clamp(x_hat, min=-1.0, max=1.0)
    def infer_reference2(self, x, steps=1, noise_fn=torch.randn_like):
        B, T, D = x.shape
        LF, HF = fft_split(x, self.split_threshold)
        LF_hat = self.diffusion_model_LF.infer(LF)
        HF_hat = self.diffusion_model_HF.sample(HF.shape, LF_hat)
        if self.augment_alpha == 'split':
            x_hat = HF_hat / self.augment_alpha + LF_hat
        else :
            x_hat = ( HF_hat - LF_hat ) / self.augment_alpha + LF_hat
        # x_hat = self.fft_reconstruct(LF_hat, HF_hat) # 拼接后的代码
        print(HF_hat.sum())
        return torch.clamp(LF_hat, min=-1.0, max=1.0), torch.clamp(x_hat, min=-1.0, max=1.0)
    def test_one_batch(self, test_batch, test_save_folder, device, epoch):
        return 
        B,T,D = test_batch.shape
        self.eval()
        y = self.infer(test_batch.to(device)).detach().cpu().numpy()
        max_idx = min(5, len(y))
        for idx in range(max_idx):
            v_gen = y[idx]
            v_true = test_batch[idx]
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            Length = v_gen.shape[0]
            Dimension = v_gen.shape[1]
            x = np.arange(1, Length + 1)

            for i in range(Dimension):
                axes[0].plot(x, v_gen[:, i], label=f'Dimension {i+1}')
            axes[0].set_title('Gen Curves')
            axes[0].set_xlabel('Length')
            axes[0].set_ylabel('Value')
            axes[0].legend(loc='upper right')

            for i in range(Dimension):
                axes[1].plot(x, v_true[:, i], label=f'Dimension {i + 1}')
            axes[1].set_title('True Curves')
            axes[1].set_xlabel('Length')
            axes[1].set_ylabel('Value')
            axes[1].legend(loc='upper right')
            save_dir = f"batch{idx}/"
            save_dir = os.path.join(test_save_folder, save_dir)
            os.makedirs(save_dir, exist_ok=True)
            save_dir = os.path.join(save_dir, f'{epoch}epoch.png')
            plt.tight_layout()
            plt.savefig(save_dir)
            plt.close()
        # print('test_one_batch_done')
        self.train() 