import torch
import torch.nn.functional as F
import math
import einops
def fft_split_2D(x, split_threshold):
    T, D = x.shape
    x_reshaped = einops.rearrange(x, 't d -> d t')
    X = torch.fft.fft(x_reshaped)
    freq_cutoff = int(split_threshold * T)
    LF = X.clone()
    HF = X.clone()
    LF[:, freq_cutoff: ] = 0
    HF[:, : freq_cutoff] = 0
    LF = torch.fft.ifft(LF).real
    HF = torch.fft.ifft(HF).real
    LF = einops.rearrange(LF, 'd t -> t d')
    HF = einops.rearrange(HF, 'd t -> t d') 
    return LF,HF
def fft_split(x, split_threshold):
    # print(f'[fft_split] x_shape {x.shape} threshold {split_threshold}')
    if split_threshold < 0:
        return nfft_split(x, -split_threshold)
    if len(x.shape) == 2:
        return fft_split_2D(x, split_threshold)
    B, T, D = x.shape
    x_reshaped = einops.rearrange(x, 'b t d -> (b d) t')
    X = torch.fft.fft(x_reshaped) 
    freq_cutoff = int(split_threshold * T)  
    LF = X.clone()
    HF = X.clone()
    LF[:, freq_cutoff:] = 0  
    HF[:, :freq_cutoff] = 0  
    LF = torch.fft.ifft(LF).real
    HF = torch.fft.ifft(HF).real
    LF = einops.rearrange(LF, '(b d) t -> b t d', b=B, d=D)
    HF = einops.rearrange(HF, '(b d) t -> b t d', b=B, d=D)
    return LF, HF
def nfft_split_2D(x, n_fft = 8, norm:bool=True):
    T, D = x.shape
    x = einops.rearrange(x, 't d -> d t')
    X = torch.stft(x, n_fft, normalized=norm, return_complex=True, window=torch.hann_window(window_length=n_fft))
    LF = X.clone()
    HF = X.clone()
    LF[1:,:,:] = 0
    HF[0,:,:] = 0
    LF = torch.istft(LF, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft))
    HF = torch.istft(HF, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft))
    LF = einops.rearrange(LF, 'd t -> t d')
    HF = einops.rearrange(HF, 'd t -> t d') 
    return LF, HF
def nfft_split(x, n_fft = 8, norm:bool=True):
    if len(x.shape) == 2:
        return nfft_split_2D(x, n_fft, norm)
    B, T, D = x.shape
    x = einops.rearrange(x, 'b t d -> (b d) t')
    X = torch.stft(x, n_fft, normalized=norm, return_complex=True, window=torch.hann_window(window_length=n_fft))
    LF = X.clone()
    HF = X.clone()
    LF[1:,:,:] = 0
    HF[0,:,:] = 0
    LF = torch.istft(LF, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft))
    HF = torch.istft(HF, n_fft, normalized=norm, window=torch.hann_window(window_length=n_fft))
    LF = einops.rearrange(LF, '(b d) t -> b t d', b=B, d=D)
    HF = einops.rearrange(HF, '(b d) t -> b t d', b=B, d=D)
    return LF, HF

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def calculate_ba_parameters(betas):
    params = {}

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

    params['betas'] = betas
    params['alphas'] = alphas
    params['alphas_cumprod'] = alphas_cumprod
    params['alphas_cumprod_prev'] = alphas_cumprod_prev
    params['sqrt_alphas_cumprod'] = torch.sqrt(alphas_cumprod)
    params['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1. - alphas_cumprod)
    params['sqrt_recip_alphas_cumprod'] = torch.sqrt(1. / alphas_cumprod)
    params['sqrt_recipm1_alphas_cumprod'] = torch.sqrt(1. / alphas_cumprod - 1)

    '''
    q(x_{t-1} | x_t, x_0) = Normal( 
        mean = posterior_q_mean_x0 * x_0 + posterior_q_mean_xt * x_t, 
        std  = posterior_variance
    )
    '''
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    params['posterior_log_variance_clipped'] = torch.log(posterior_variance.clamp(min=1e-20))
    params['posterior_q_mean_x0'] = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
    params['posterior_q_mean_xt'] = torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    '''
    p(x_{t-1} | x_t) = Normal(
        mean = posterior_p_mean_xt * x_t + posterior_p_mean_noise * MODEL(x_t, t)
        std = posterior_variance
    )
    '''
    params['posterior_p_mean_xt'] = 1. / torch.sqrt(alphas)
    params['posterior_p_mean_noise'] = - betas / (torch.sqrt(alphas) * params['sqrt_one_minus_alphas_cumprod'])

    params['loss_weight'] = torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100
    return params


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))