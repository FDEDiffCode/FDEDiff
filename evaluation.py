import os
import sys
import json
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


from metric.utils import (
    display_scores,
    load_npz,
    load_npy,
    setseed
)

from metric.cross_correlation import Cross_CorrScore, Auto_CorrScore
from metric.context_fid import Context_FID
from metric.ds_metric import discriminative_score_metrics
from metric.ps_metric import predictive_score_metrics

# 获取命令行参数
if len(sys.argv) != 4:
    print("Usage: python script.py <lab_name> <oridata_path> <gen_path>")
    sys.exit(1)

lab_name = sys.argv[1]
oridata_path = sys.argv[2]
gen_path = sys.argv[3]

# 设置随机种子
setseed()

# 设置日志记录
log_dir = f'./results/logs/{lab_name}'
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'log.txt')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler(sys.stdout)
])

# 加载数据
logging.info(f"Loading data from {oridata_path} and {gen_path}")
if oridata_path[-1] == 'z':
    ori_data, gen_data = load_npz('', origin_path=oridata_path, gen_path=gen_path)
else :
    ori_data, gen_data = load_npy('', origin_path=oridata_path, gen_path=gen_path)
logging.info(f"Original data shape: {ori_data.shape}, Generated data shape: {gen_data.shape}")
if gen_data.shape[0] < ori_data.shape[0]:
    padding_size = ori_data.shape[0] - gen_data.shape[0]
    padded_gen_data = np.zeros((ori_data.shape[0], gen_data.shape[1], gen_data.shape[2]))
    padded_gen_data[:gen_data.shape[0], :, :] = gen_data
    gen_data = padded_gen_data
def normalize_if_needed(data, data_type):
    data_min, data_max = data.min(), data.max()
    # 判断是否需要归一化
    if data_min < 0 or data_max > 1:
        logging.info(f"{data_type} data need norm {data_min} {data_max}")
        scaler = MinMaxScaler(feature_range=(0, 1))
        L, W, D = data.shape
        data_re = data.reshape(-1, D)
        data_norm = scaler.fit_transform(data_re)
        data_norm = data_norm.reshape(L*W, D)
        x = []
        for i in range(0, L*W, 1):
            start = i
            end = i + window
            if end > L*W:
                break 
            x.append(data_norm[start:end, :])
        return x
    else:
        return data

# ori_data = normalize_if_needed(ori_data, 'ori')
# gen_data = normalize_if_needed(gen_data, 'gen')
ori_data_min, ori_data_max = ori_data.min(), ori_data.max()
gen_data_min, gen_data_max = gen_data.min(), gen_data.max()
logging.info(f"ori {ori_data_min} {ori_data_max} || gen {gen_data_min} {gen_data_max}")
# 确保数据已经被正则化到 [0, 1]
# assert 0 <= ori_data_min <= 1 and 0 <= ori_data_max <= 1, "Original data not normalized to [0, 1]"
# assert 0 <= gen_data_min <= 1 and 0 <= gen_data_max <= 1, "Generated data not normalized to [0, 1]"


# 初始化结果字典
results = {"name": lab_name}

# 计算 CFID
logging.info("Solving CFID...")
cfid_list = []
for i in range(5):
    cfid = Context_FID(ori_data, gen_data)
    cfid_list.append(cfid)
    logging.info(f'{i}-iter cfid: {cfid}\n')

results['cfid'] = display_scores(cfid_list)

# 计算 CACD 和 ACD
logging.info("Solving CACD and ACD...")
cacd = Cross_CorrScore(ori_data, gen_data)
acd = Auto_CorrScore(ori_data, gen_data, 5)

results['cacd'] = cacd
results['acd'] = acd

# 计算 DS
logging.info("Solving DS...")
ds_list = []
for i in range(5):
    ds_score = discriminative_score_metrics(ori_data, gen_data)
    ds_list.append(ds_score)
    logging.info(f'{i}-iter ds: {ds_score}\n')

results['ds'] = display_scores(ds_list)


logging.info("Solving PS...")
ps_list = []
for i in range(5):
    ps_score = predictive_score_metrics(ori_data, gen_data)
    ps_list.append(ps_score)
    logging.info(f'{i}-iter ps: {ps_score}\n')

results['ps'] = display_scores(ps_list)


output_dir = f'./results/{lab_name}'
os.makedirs(output_dir, exist_ok=True)

result_file = os.path.join(output_dir, 'result.json')


with open(result_file, 'w') as f:
    json.dump(results, f, indent=4)

logging.info(f"Results saved to {result_file}")
