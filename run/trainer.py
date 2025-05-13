import os
import time
import json
import shutil
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from runs.lr_sch import WarmUpSch
from runs.plt_show import plot_loss_curve  # 导入绘图函数

class Trainer:
    def __init__(self, config, model, dataloader, device, test_dataloader=None):
        self.model = model
        self.device = device
        self.dataloader = dataloader
        self.n_epochs = config['max_epochs']
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), config['lr'])
        self.log_dir = os.path.join('experiments', self.config["lab_name"])
        self.test_dataloader = test_dataloader
        #config['sch']['base_lr'] = config['lr']
        self.scheduler = WarmUpSch(self.optimizer, **config['sch'])


    def save_train_result(self, running_loss, total_runtime, total_samples_per_second, total_steps_per_second, learning_rate, step, epoch, result_dir=None):
        train_result = {
            'final_loss' : running_loss,
            'final_lr' : learning_rate, 
            'final_step' : step,
            'final_epoch' : epoch,
            'total_runtime': total_runtime,
            'total_samples_per_second': total_samples_per_second,
            'total_steps_per_second': total_steps_per_second
        }
        if result_dir == None : result_dir = self.log_dir
        with open(os.path.join(result_dir, 'train_result.json'), 'w') as f:
            json.dump(train_result, f, indent=4)

    def save_training_info(self, epoch, step, running_loss, current_lr, epoch_runtime, epoch_samples, log_file):
        samples_per_second_epoch = epoch_samples / epoch_runtime
        steps_per_second_epoch = step / epoch_runtime

        log_entry = {
            'epoch': epoch,
            'loss': running_loss,
            'learning_rate': current_lr,
            'runtime': epoch_runtime,
            'samples_per_second': samples_per_second_epoch,
            'steps_per_second': steps_per_second_epoch
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        checkpoint_dir = os.path.join(self.log_dir, f'checkpoint-{epoch}epoch')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_model_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(self.model.state_dict(), checkpoint_model_path)

        self.save_train_result(running_loss / step, epoch_runtime, samples_per_second_epoch, steps_per_second_epoch, current_lr, step, epoch, checkpoint_dir)



    def train(self):
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir, 'train_log.jsonl')
        model_save_path = os.path.join(self.log_dir, 'model.pth')
        all_loss = []

        pbar = tqdm(range(self.n_epochs), desc='training')
        
        start_time = time.time()
        all_step = 0
        test_epoch = max(self.n_epochs // 3, 1)
        vali_batch = None

        for epoch in pbar:
            self.model.train()
            epoch_start_time = time.time()
            step = 0
            running_loss = 0.0
            epoch_samples = 0
            current_lr = self.scheduler.get_last_lr()[0]

            # 这里是用来显示进度条的，显示的内容包括epoch和step
            pbar.set_postfix(epoch=epoch, step=step, loss='N/A', lr=current_lr)

            for x_batch in self.dataloader:
                if vali_batch is None: vali_batch = x_batch
                x_batch = x_batch.to(self.device)
                loss = self.model(x_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                step += 1
                all_step += 1
                epoch_samples += x_batch.size(0)

                self.scheduler.step_iter()

                # 更新进度条的显示，显示epoch, step和当前的loss
                pbar.set_postfix(epoch=epoch, step=step, loss=running_loss / (step + 1), lr=current_lr)

            self.scheduler.step(running_loss / step)

            # 保存模型checkpoint
            if (epoch + 1) % self.config['save_per_epoch'] == 0:
                self.save_training_info(epoch, step, running_loss, current_lr, time.time() - epoch_start_time, epoch_samples, log_file)

            # 绘制验证或测试阶段的图
            if epoch % test_epoch == 0 or epoch == self.n_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    if vali_batch is not None:
                        
                        test_dir = os.path.join(self.log_dir, 'test_epoch/valid/')
                        self.model.test_one_batch(vali_batch, test_dir, self.device, epoch)
                    if self.test_dataloader is not None:
                        test_dir = os.path.join(self.log_dir, 'test_epoch/test/')
                        os.makedirs(test_dir, exist_ok=True)
                        for test_data_batch in self.test_dataloader:
                            self.model.test_one_batch(test_data_batch, test_dir, self.device, epoch)
                            break

            running_loss /= step
            all_loss.append(running_loss)

            epoch_end_time = time.time()
            epoch_runtime = epoch_end_time - epoch_start_time

            # writer.add_scalar('Loss/train', running_loss, epoch)
            # 在每个epoch结束时更新进度条
            pbar.set_postfix(epoch=epoch, step=step, loss=running_loss, lr=current_lr)

        torch.save(self.model.state_dict(), model_save_path)
        total_runtime = time.time() - start_time
        total_samples_per_second = epoch_samples / total_runtime
        total_steps_per_second = sum([x_batch.size(0) for x_batch in self.dataloader]) / total_runtime
        self.save_train_result(running_loss, total_runtime, total_samples_per_second, total_steps_per_second, current_lr, step, self.n_epochs)

        writer.close()

        plot_loss_curve(all_loss, self.n_epochs, self.log_dir)
