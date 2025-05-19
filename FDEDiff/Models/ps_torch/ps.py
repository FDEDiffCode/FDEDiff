import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
import numpy as np

from tqdm import tqdm

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.ac = nn.Sigmoid()
    
    def forward(self, x):
        # x (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = self.fc(out)
        return self.ac(out)

def eval_ps(ori_data, gen_data, device, n_epochs):
    N, L, D = ori_data.shape
    batch_size = 16
    output_dim = 1

    model = GRUNetwork(D - 1, D // 2, num_layers=2, output_size=output_dim).to(device)
    criterion = nn.L1Loss() 
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    ori_data_tensor = torch.tensor(ori_data, dtype=torch.float32)
    generated_data_tensor = torch.tensor(gen_data, dtype=torch.float32)

    train_dataset = TensorDataset(generated_data_tensor[:, :-1, :-1], generated_data_tensor[:, 1:, -1:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(ori_data_tensor[:, :-1, :-1], ori_data_tensor[:, 1:, -1:])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    pbar = tqdm(range(n_epochs), desc='training')
    model.train()

    for E in pbar:
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        running_loss /= len(train_loader)
        pbar.set_postfix(loss=running_loss)
    
    model.eval()
    total_mae = 0
    count = 0

    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            outputs = outputs.view(-1)
            y_batch = y_batch.view(-1)
            mae = mean_absolute_error(y_batch.cpu().numpy(), outputs.cpu().numpy())
            total_mae += mae * x_batch.size(0)
            count += x_batch.size(0)
    total_mae /= count

    return total_mae