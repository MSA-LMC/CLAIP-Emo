import json, os
import matplotlib.pyplot as plt

# 日志路径
# log_file = '/home/u2022111029/project/DFER_AV_explore/saved/dfew/audio_video/clip+clap/MMDFER_CLIP_b32_SwinTSmall_Baseline/eval_split01_lr_1e-5_epoch_100_bs_64_size224_a1024_sr4_server170/log.txt'
def plt_logs(path):
    # 初始化指标列表
    log_file = os.path.join(path, 'log.txt')
    epochs = []
    lr = []

    losses = {}
    metrics = {}
    
    # 读取并解析每一行 JSON
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            data = json.loads(line)
            for key, value in data.items():
                if key == 'epoch':
                    epochs.append(data['epoch'])
                elif key == 'train_lr':
                    lr.append(data['train_lr'] * 1e4)  # 放大1e4方便画图
                elif key.endswith('_loss'):
                    if key not in losses:
                        losses[key] = []
                    losses[key].append(value)
                elif key.endswith('_acc1') or key.endswith('_acc'):
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
           


    # 绘图
    plt.figure(figsize=(12, 6))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    for key, value in losses.items():
        plt.plot(epochs, value, label=key)
    plt.plot(epochs, lr, label='lrx1e5', linestyle='-.')
    plt.xlabel('Epoch')
    plt.xlim([0, 100])
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    # 
    plt.legend(loc='upper left')
    plt.grid(True)

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    for key, value in metrics.items():
        plt.plot(epochs, value, label=key)

    plt.xlabel('Epoch')
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')

    # find best val_acc1 and its epoch
    try:
        best_val_acc1 = max(metrics.get('val_acc1', []))
        best_epoch = epochs[metrics.get('val_acc1', []).index(best_val_acc1)]
        plt.axhline(y=best_val_acc1, color='r', linestyle='--', label=f'Best val_acc1: {best_val_acc1:.2f}% at epoch {best_epoch}')
    except:
        print("No val_acc1 found in metrics.")
     
    try:
        best_val_acc1 = max(metrics.get('sfer_val_sfer_acc1', []))
        best_epoch = epochs[metrics.get('sfer_val_sfer_acc1', []).index(best_val_acc1)]
        plt.axhline(y=best_val_acc1, color='r', linestyle='--', label=f'Best SFER val_acc1: {best_val_acc1:.2f}% at epoch {best_epoch}')
    except:
        print("No val_acc1 found in metrics.")
        
     
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'training_log.png'), dpi=300)
    plt.close()
    
