import matplotlib.pyplot as plt # 必须导入用于绘图

def plot_training_history(history):
    """绘制训练记录图表"""
    epochs = range(len(history['train_loss']))
    
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train Loss')
    #plt.plot(epochs, history['val_loss'], 'b-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    #plt.plot(epochs, history['train_acc'], 'r-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'b-', label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png') # 自动保存图片到当前目录
    
    #plt.show()