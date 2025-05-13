import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import os
import time
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
class VQVAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        vq,
    ):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vqvae = vq

    def Fourier_Loss(self, f, f_hat):  # f (B, L, input_dim)
        F = torch.fft.fft(f, dim=1)
        F_hat = torch.fft.fft(f_hat, dim=1)
        diff = F - F_hat
        magnitude_diff = torch.abs(diff)
        return torch.mean(magnitude_diff)

    def forward(self, x, lr=None):  # (B, L, input_dim)
        z = self.encoder(x)  # (B, L / patch_len, output_dim)
        zq, code_loss = self.vqvae(z)  # (B, L / patch_len, output_dim)
        x_recon = self.decoder(zq)  # (B, L, input_dim)
        # recon_loss = F.mse_loss(x_recon, x)
        recon_loss1 = F.mse_loss(x_recon, x)
        recon_loss2 = self.Fourier_Loss(
            x, x_recon)/math.sqrt(x_recon.shape[1])*0.5
        recon_loss = recon_loss1 + recon_loss2
        print(
            f'\rfft_loss {recon_loss2:.4f}, fse_loss {recon_loss1:.4f}, code_loss {code_loss:.4f} len {x_recon.shape[1]} LR {lr}',
            end='',
            flush=True
        )
        loss = recon_loss + code_loss
        return loss

    def infer(self, x):  # (B, L, input_dim)
        z = self.encoder(x)
        zq, _ = self.vqvae(z)
        return zq  # (B, L / patch_len, output_dim)

    def decode(self, zq):  # (B, L / patch_len, output_dim)
        return self.decoder(zq)  # (B, L, input_dim)

    def test_one_batch(self, test_batch, test_save_folder, device, epoch):
        self.eval()
        y = self.infer(test_batch.to(device))
        t_y = self.decode(y).detach().cpu().numpy()
        
        max_idx = min(9, len(y))
        for idx in range(max_idx):
            v_gen = t_y[idx]
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