import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.transformer.encoder import TimeSeriesEncoder, TimeSeriesDecoder, TimeSeriesEncoderWithCond
from Models.vqvae.vq import VectorQuantizer
from Models.vqvae.vae import VQVAE

from Models.ddpm.conv_predictor import ConvPredictor
from Models.ddpm.predictor import MLP_noise
from Models.ddpm.diffusion import Diffusion
from Models.ddpm.biddpm import Biddpm

def build_transformer_vqvae(config):
    cfg = config['model']
    base_cfg = cfg['common']
    encoder = TimeSeriesEncoder(
        base_cfg['input_dim'],
        base_cfg['d_model'],
        base_cfg['output_dim'],
        patch_len=base_cfg['patch_len'],
        **cfg['encoder']
    )
    decoder = TimeSeriesDecoder(
        base_cfg['output_dim'],
        base_cfg['d_model'],
        base_cfg['input_dim'],
        patch_len=base_cfg['patch_len'],
        **cfg['decoder']
    )
    vq = VectorQuantizer(
        embedding_dim=base_cfg['output_dim'],
        **cfg['vq']
    )
    return VQVAE(encoder, decoder, vq).float()


def build_ddpm(config):
    cfg = config['model']
    base_cfg = cfg['common']
    noise_predictor = MLP_noise(
        base_cfg['input_len'],
        base_cfg['input_dim'],
        **cfg['mlp_noise']
    )
    return Diffusion(
        noise_predictor, 
        **cfg['ddpm']
    )

def build_conv_ddpm(config):
    cfg = config['model']
    base_cfg = cfg['common']
    noise_predictor = ConvPredictor(
        base_cfg['input_dim'],
        **cfg['conv']
    )
    return Diffusion(
        noise_predictor,
        **cfg['ddpm']
    )

def build_transformer_ddpm(config):
    cfg = config['model']
    base_cfg = cfg['common']
    noise_predictor = TimeSeriesEncoder(
        input_dim=base_cfg['input_dim'],
        output_dim=base_cfg['input_dim'],
        **cfg['encoder']
    )
    return Diffusion(
        noise_predictor,
        **cfg['ddpm']
    )

def build_transformer_ddpm_with_cond(config):
    cfg = config['model']
    base_cfg = cfg['common']
    noise_predictor = TimeSeriesEncoderWithCond(
        input_dim=base_cfg['input_dim'],
        output_dim=base_cfg['input_dim'],
        **cfg['encoder']
    )
    return Diffusion(
        noise_predictor,
        **cfg['ddpm']
    )

def build_biddpm(config):
    cfg = config['model']
    diffusion_model_LF = build_transformer_ddpm(config)
    diffusion_model_HF = build_transformer_ddpm_with_cond(config)    
    return Biddpm(
        diffusion_model_LF,
        diffusion_model_HF,
        **cfg['biddpm']
    )