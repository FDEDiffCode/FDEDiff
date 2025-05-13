import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

import os

def train_test_divide(ori_data, gen_data, train_rate=0.8):

    def single_divide(data, label, train_rate):
        n = len(data)
        idx = np.random.permutation(n)
        train_idx = idx[ : int(n * train_rate)]
        test_idx = idx[int(n * train_rate) : ]

        train_x = [data[i] for i in train_idx]
        test_x = [data[i] for i in test_idx]
        train_y = [label for i in train_idx]
        test_y = [label for i in test_idx]

        return train_x, train_y, test_x, test_y

    def merge(ori_list, gen_list):
        res = ori_list
        res.extend(gen_list)
        return np.array(res)

    ori_train_x, ori_train_y, ori_test_x, ori_test_y = single_divide(ori_data, 1, train_rate)
    gen_train_x, gen_train_y, gen_test_x, gen_test_y = single_divide(gen_data, 0, train_rate)

    train_x = merge(ori_train_x, gen_train_x)
    train_y = merge(ori_train_y, gen_train_y)
    test_x = merge(ori_test_x, gen_test_x)
    test_y = merge(ori_test_y, gen_test_y)

    return train_x, train_y, test_x, test_y

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_dataloader(data, labels, batch_size):
    data_tensor = torch.tensor(data).to(torch.float)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(data_tensor, labels_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

def eval_ds(ori_data, gen_data, device, n_epochs):
    # ori_data, gen_data: ndarray, (batch, len, D)
    # config
    N, L, D = ori_data.shape
    batch_size = 16
    model = GRUNetwork(D, D // 2, num_layers=2, output_size=2).to(device)
    # config over

    train_x, train_y, test_x, test_y = train_test_divide(ori_data, gen_data)
    train_loader = make_dataloader(train_x, train_y, batch_size)
    test_loader = make_dataloader(test_x, test_y, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_log = []
    pbar = tqdm(range(n_epochs), desc='training')
    model.train()
    
    for E in pbar:
        running_loss = 0.0
        accuracy = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()  
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accuracy += ((predicted == labels).sum().item())
        running_loss /= len(train_loader)
        accuracy /= len(train_x)
        pbar.set_postfix(loss=running_loss, acc=accuracy)
        loss_log.append(running_loss)
    
    model.eval()
    accuracy = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            accuracy += ((predicted == labels).sum().item())
    accuracy /= len(test_x)

    return np.abs(accuracy - 0.5)