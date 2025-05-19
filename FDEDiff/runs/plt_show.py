# plt_show.py
import matplotlib.pyplot as plt
import os
def plot_loss_curve(all_loss, n_epochs, log_dir):
    """
    绘制训练损失曲线并保存为 PNG 文件。
    :param all_loss: 训练过程中每个 epoch 的损失
    :param n_epochs: 总训练轮次
    :param log_dir: 日志目录，保存图像文件
    """
    plt.figure()
    plt.plot(range(n_epochs), all_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))
    plt.close()

    start_idx = int(n_epochs * 3 / 4)
    end_idx = n_epochs
    loss_last_quarter = all_loss[start_idx:end_idx]
    epochs_last_quarter = range(start_idx, end_idx)
    plt.figure()
    plt.plot(epochs_last_quarter, loss_last_quarter, label='Train Loss (Last 1/4)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Last 1/4)')
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'loss_curve_last_quarter.png'))
    plt.close()
