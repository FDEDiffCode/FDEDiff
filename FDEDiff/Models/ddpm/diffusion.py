import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from Models.ddpm.utils import (
    linear_beta_schedule, 
    cosine_beta_schedule,
    calculate_ba_parameters,
    extract
) 


class Diffusion(nn.Module):
    def __init__(
        self,
        step_predictor,
        timesteps = 1000,
        beta_schedule = 'cosine',
        predict_target = 'eps'
    ):
        super(Diffusion, self).__init__()
        self.model = step_predictor

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps)
        self.num_timesteps = int(betas.shape[0])
        params = calculate_ba_parameters(betas)
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        for name, val in params.items():
            register_buffer(name, val)
        self.loss_fn = F.mse_loss
        self.predict_target = predict_target
    
    def get_noise_given_x0_xt(self, x0, xt, t):
        # noise = (xt - sqrt(alpha_cum) * x0) / sqrt(1 - alpha_cum)
        return (
            (xt - extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0) /
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        )

    def q_xt_given_x0(self, x0, t, noise):
        # xt = noise * sqrt(1 - alpha_cum) + sqrt(alpha_cum) * x0
        return (
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise + 
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
        )
    
    def q_v_given_x0(self, x0, t, noise):
        # xt = noise * sqrt(1 - alpha_cum) + sqrt(alpha_cum) * x0
        # v = d(xt)/d(phi(t)) = -sqrt(1 - alpha_cum) * x0 + sqrt(alpha_cum) * noise
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * x0
        )
    
    def p_noise_given_v(self, xt, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, xt.shape) * v +
            extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape) * xt
        )
    
    def p_xprev_given_xt(self, xt, t):
        batched_t = torch.full((xt.shape[0],), t, device=xt.device, dtype=torch.long)
        noise = torch.randn_like(xt) if t > 0 else 0.
        log_std = extract(self.posterior_log_variance_clipped, batched_t, xt.shape)
        output = self.model(xt, batched_t)
        if self.predict_target == 'x0':
            return (
                extract(self.posterior_q_mean_x0, batched_t, xt.shape) * output +
                extract(self.posterior_q_mean_xt, batched_t, xt.shape) * xt +
                (0.5 * log_std).exp() * noise
            )
        if self.predict_target == 'v':
            output = self.p_noise_given_v(xt, batched_t, output)
        return (
            extract(self.posterior_p_mean_xt, batched_t, xt.shape) * xt +
            extract(self.posterior_p_mean_noise, batched_t, xt.shape) * output +
            (0.5 * log_std).exp() * noise
        )

    def p_loss(self, x0, t, noise, cond=None):
        xt = self.q_xt_given_x0(x0, t, noise)
        if cond == None:
            output = self.model(xt, t)
        else :
            output = self.model(xt, t, cond)
        if self.predict_target == 'eps':
            target = noise
        elif self.predict_target == 'x0':
            target = x0
        elif self.predict_target == 'v':
            target = self.q_v_given_x0(x0, t, noise)
        else:
            raise NotImplementedError(f'DDPM predict target: {self.predict_target} not supported!')
        loss = self.loss_fn(output, target)
        return loss

    def sample(self, shape):
        device = self.betas.device
        xt = torch.randn(shape, device=device)
        for t in reversed(range(0, self.num_timesteps)):
            xt = self.p_xprev_given_xt(xt, t)
        return xt

    def forward(self, x0, cond=None):
        B = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x0.device).long()
        noise = torch.randn_like(x0)
        return self.p_loss(x0, t, noise, cond)

    def infer(self, x_sample):
        sp = x_sample.shape
        xt = self.sample(sp)
        return torch.clamp(xt, min=-1.0, max=1.0)
    
    def debug_back(self, x_shape, step, save_dir, T): # x_shape = (B, L, D)
        ''' 
            生成shape为x_shape的samples
            每step步输出一次
            倒数T步逐步输出
        '''
        def save_batch_mts(x, save_dir, step): # tensor 
            import os 
            import matplotlib.pyplot as plt
            x = x.detach().cpu().numpy()
            B, L, D = x.shape
            for i in range(B):
                dir_path = os.path.join(save_dir, 'backward', str(i))
                os.makedirs(dir_path, exist_ok=True)
                plt.figure(figsize=(10, 10))
                for j in range(D):
                    plt.plot(x[i, :, j], label=str(j))
                plt.legend()
                plt.title(f'backward_step_{step}')
                plt_path = os.path.join(dir_path, f'{step}.png')
                plt.savefig(plt_path)
                plt.close()
        
        now = 0
        B = x_shape[0]
        xt = torch.randn(size=x_shape, device=self.betas.device)
        for i in tqdm(reversed(range(0, self.num_timesteps))):
            xt = self.p_xprev_given_xt(xt, i)
            now += 1
            if now % step == 0 or i <= T:
                save_batch_mts(xt, save_dir, now)
        
    def test_one_batch(self, test_batch, test_save_folder, device, epoch):
        self.debug_back(test_batch.shape, 50, test_save_folder, 50)
        # pass
