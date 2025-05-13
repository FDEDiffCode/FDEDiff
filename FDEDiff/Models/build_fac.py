from Models.build import *

def build_model_from_config(config):
    name = config['model']['name']
    if name == 'build_transformer_vqvae':
        return build_transformer_vqvae(config)
    elif name == 'build_ddpm':
        return build_ddpm(config)
    elif name == 'build_conv_ddpm':
        return build_conv_ddpm(config)
    elif name == 'build_transformer_ddpm':
        return build_transformer_ddpm(config)
    elif name == 'build_biddpm':
        return build_biddpm(config)
    else:
        raise NotImplementedError(f'Build model {name} not support!')
