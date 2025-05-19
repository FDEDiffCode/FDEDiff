import os,sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import datetime
from FDEDiff.runs.trainer import Trainer
from FDEDiff.metric.utils import setseed, load_yaml_config
from FDEDiff.dataloader.build import build_dataloader,build_dataloader_LF
from FDEDiff.Models.build_fac import build_model_from_config
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
def parse_args():
    parser = argparse.ArgumentParser(description='tsg pipeline')
    parser.add_argument('--config', type=str, default='./FDEDiff/config/biddpm_test.yml')
    parser.add_argument('--window', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--seed', type=int, default=9163)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--lf_path', type=str, default=None)
    args = parser.parse_args()
    return args
data_loadpath = {
    'etth' : {
        'loadpath' : './data/ETTh.csv',
        'dim' : 7,
        'keep_first' : False
    },
    'energy' : {
        'loadpath' : './data/energy_data.csv',
        'dim' : 28,
        'keep_first' : True
    },
    'stock' : {
        'loadpath' : './data/stock.csv',
        'dim' : 6,
        'keep_first' : True
    },
    'electricity' : {
        'loadpath' : './data/electricity.csv',
        'keep_first' : False,
        'dim' : 321
    },
    'traffic' : {
        'loadpath' : './data/traffic.csv',
        'keep_first' : True,
        'dim' : ''
    },
    'fmri' : {
        'loadpath' : './data/fmri.csv',
        'keep_first' : True,
        'dim' : ''        
    },
    'electricity_pca_30' : {
        'loadpath' : './data/electricity_pca_30.csv',
        'keep_first' : True,
        'dim' : ''        
    },
    'traffic_pca_30' : {
        'loadpath' : './data/traffic_pca_30.csv',
        'keep_first' : True,
        'dim' : ''        
    }
}
def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))

def load_model(model, save_dir):
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model.pth'), weights_only=True))
    return model

def save_loss(loss_logs, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.plot(loss_logs, label='Loss', color='blue')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path)
    plt.close()

import os
import subprocess

def run_evaluation(lab_name, oridata_path, gen_path):
    # 定义要执行的命令
    command = [
        'python', 'evaluation.py',  # 假设评测代码文件名为 evaluation.py
        lab_name,                   # 实验名称
        oridata_path,               # 原始数据路径
        gen_path                    # 生成数据路径
    ]
    # 调用评测程序并捕获输出
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode == 0:
        print("Evaluation completed successfully.")
        print(result.stdout.decode())  # 打印标准输出
    else:
        print("Error occurred during evaluation.")
        print(result.stderr.decode())  # 打印错误输出
import sys
if __name__ == '__main__':
    args = parse_args()
    setseed(args.seed)
    config = load_yaml_config(args.config)
    if args.window != None:
        config['dataloader']['dataset']['window'] = args.window
        config['model']['common']['input_len'] = args.window
    if args.dataset != None:
        dataset_name = args.dataset
    else :
        dataset_name = config['dataloader']['dataset']['name']
    config['dataloader']['dataset']['name'] = dataset_name
    config['dataloader']['dataset']['data_root'] = data_loadpath[dataset_name]['loadpath']
    config['dataloader']['dataset']['keep_first'] = data_loadpath[dataset_name]['keep_first']
    import pandas as pd
    df = pd.read_csv(data_loadpath[dataset_name]['loadpath'])
    feature_size = df.shape[1]
    if config['dataloader']['dataset']['keep_first'] == False:
        feature_size -= 1
    config['dataloader']['dim'] = feature_size
    #config['dataloader']['dim'] = data_loadpath[args.dataset]['dim']
    config['model']['common']['input_dim'] = feature_size
    if args.lf_path != None:
        config['exp']['biddpm']['ddpm_lf']['load_path'] = args.lf_path
    # WARMUP--------------------------------------------------------------
    if 'warmup_ratio' in config['exp']['biddpm']['ddpm_lf']['sch']:
        warmup_ratio_lf = config['exp']['biddpm']['ddpm_lf']['sch']['warmup_ratio']
    else :
        warmup_ratio_lf = 0.005
    if 'warmup_ratio' in config['exp']['biddpm']['ddpm_hf']['sch']:
        warmup_ratio_hf = config['exp']['biddpm']['ddpm_hf']['sch']['warmup_ratio']
    else :
        warmup_ratio_hf = 0.005
    epoch_steps =  df.shape[0] / config['dataloader']['batch_size']
    config['exp']['biddpm']['ddpm_lf']['sch']['warmup_steps'] = int(epoch_steps * config['exp']['biddpm']['ddpm_lf']['max_epochs'] * warmup_ratio_lf)
    config['exp']['biddpm']['ddpm_hf']['sch']['warmup_steps'] = int(epoch_steps * config['exp']['biddpm']['ddpm_hf']['max_epochs'] * warmup_ratio_hf)
    config['exp']['biddpm']['threshold'] = config['model']['biddpm']['split_threshold']
    dataloader, dataset = build_dataloader(config)
    dataloader_lf, dataset_lf = build_dataloader_LF(config, config['exp']['biddpm']['threshold'])
    test_dataloader, test_dataset = build_dataloader(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config['exp']['lab_name'] = f"{config['dataloader']['dataset']['name']}/{config['exp']['lab_name']}/W{config['dataloader']['dataset']['window']}_{current_date}"

    ddpm_model = build_model_from_config(config)
    config['exp']['biddpm']['ddpm_lf']['lab_name'] = f'{config["exp"]["lab_name"]}/lf'
    config['exp']['biddpm']['ddpm_hf']['lab_name'] = f'{config["exp"]["lab_name"]}/hf'
    save_dir = f"./experiments/{config['exp']['lab_name']}"
    os.makedirs(os.path.join(save_dir, 'config'), exist_ok=True)
    yaml_save_dir = os.path.join(save_dir, 'config/config.yml')
    yaml.dump(config, open(yaml_save_dir, 'w'), default_flow_style=False, allow_unicode=True)
    print(f'config saving at {yaml_save_dir}')
    if 'load_path' not in config['exp']:
        ddpm_model.to(device)
        if 'load_path' not in config['exp']['biddpm']['ddpm_hf']:
            if 'load_path' not in config['exp']['biddpm']['ddpm_lf']:
                trainer_lf = Trainer(
                    config['exp']['biddpm']['ddpm_lf'],
                    ddpm_model.diffusion_model_LF,
                    dataloader_lf,
                    device
                )
                trainer_lf.train()
                config['exp']['biddpm']['ddpm_lf']['load_path'] = f"./experiments/{config['exp']['lab_name']}/lf/model.pth"
                if os.path.exists(f"./experiments/{config['exp']['lab_name']}/lf/model_opt.pth"):
                    ddpm_model.diffusion_model_LF.load_state_dict(torch.load(f"./experiments/{config['exp']['lab_name']}/lf/model_opt.pth"))
            else :
                ddpm_model.diffusion_model_LF.load_state_dict(torch.load(config['exp']['biddpm']['ddpm_lf']['load_path']))
            ddpm_model.diffusion_model_LF.eval()
            for param in ddpm_model.diffusion_model_LF.parameters():
                param.requires_grad = False
        
            trainer_hf = Trainer(
                config['exp']['biddpm']['ddpm_hf'],
                ddpm_model,
                dataloader,
                device
            )
            trainer_hf.train()
            config['exp']['biddpm']['ddpm_hf']['load_path'] = f"./experiments/{config['exp']['lab_name']}/hf/model.pth"
            if os.path.exists(f"./experiments/{config['exp']['lab_name']}/hf/model_opt.pth"):
                    ddpm_model.load_state_dict(torch.load(f"./experiments/{config['exp']['lab_name']}/hf/model_opt.pth"))
        else :
            ddpm_model.load_state_dict(torch.load(config['exp']['biddpm']['ddpm_hf']['load_path']))
        ddpm_model.eval()
        for param in ddpm_model.parameters():
            param.requires_grad = False
        eval_result = []
        align_origin = []
        with torch.no_grad():
            for i in tqdm(test_dataloader):
                x_i = torch.tensor(i, dtype=torch.float32).to(device)  # 转换为 Tensor 并移动到设备
                y_i = ddpm_model.infer(x_i).detach().cpu().numpy() 
                eval_result.append(y_i)
        eval_result_tensor = torch.tensor(eval_result, dtype=torch.float32)
        eval_result_numpy = eval_result_tensor.numpy()
        *_, L, D = eval_result_tensor.shape
        np.save(os.path.join(save_dir, f'gen.npy'), ((eval_result_numpy+1)*0.5).reshape(-1, L, D))
        config['exp']['load_path'] = os.path.join(save_dir, f'gen.npy')
        yaml.dump(config, open(yaml_save_dir, 'w'), default_flow_style=False, allow_unicode=True)
    else :
        eval_result_numpy = np.load(config['exp']['load_path'])
        eval_result_numpy = (eval_result_numpy * 2) - 1
        *_, L, D = eval_result_numpy.shape
        np.save(os.path.join(save_dir, f'gen.npy'), ((eval_result_numpy + 1) * 0.5).reshape(-1, L, D))
        config['exp']['load_path'] = os.path.join(save_dir, f'gen.npy')
        yaml.dump(config, open(yaml_save_dir, 'w'), default_flow_style=False, allow_unicode=True)
    eval_result = eval_result_numpy.reshape(-1, L, D)
    test_dataset_result = torch.stack([torch.tensor(j, device='cuda') for j in test_dataset.train_arrs]).cuda()
    align_origin = []
    for i in tqdm(eval_result):
        try:
            y_i = torch.tensor(i, device='cuda').unsqueeze(0)
            fse_matrix = torch.sum((test_dataset_result - torch.tensor(y_i, device='cuda'))**2, dim=(1, 2))
            align_origin.append(test_dataset_result[torch.argmin(fse_matrix)].cpu().numpy())
            del fse_matrix
        except Exception as e:
            print(f'shape:i {i.shape} yi {y_i.shape} j {test_dataset_result.shape} fse {fse_matrix.shape} \n {e} \n')
            sys.exit()
    align_origin = torch.tensor(align_origin, dtype=torch.float32).numpy()
    
    np.save(os.path.join(save_dir, f'origin.npy'), test_dataset.arrs.reshape(-1, L, D))
    np.save(os.path.join(save_dir, f'origin_align.npy'), ((align_origin + 1)*0.5).reshape(-1, L, D))
    


    run_evaluation(f"{config['exp']['lab_name']}/valid", os.path.join(save_dir, f'origin_align.npy'), os.path.join(save_dir, f'gen.npy'))
    #run_evaluation(f"{config['exp']['lab_name']}/test", os.path.join(save_dir, f'origin.npy'), os.path.join(save_dir, f'gen.npy'))
    run_evaluation(config['exp']['lab_name'], os.path.join(save_dir, f'origin.npy'), os.path.join(save_dir, f'gen.npy'))
    yaml.dump(config, open(yaml_save_dir, 'w'), default_flow_style=False, allow_unicode=True)
    print(f'yaml save at {yaml_save_dir}')