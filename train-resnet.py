import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.model import get_resnet50_classifier # 确保你已创建此模型文件
from dataloader import get_data_loaders
from others.plot import plot_training_history

# --- 实验配置（保持与 VGG 实验一致以保证公平对比） ---
DATA_DIR = '/public/home/liuquanlong_gsc/Datasets/Dogvscat/'
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 7
counter = 0
best_val_loss = float('inf') # 记录目前见过的最低验证损失



def train_resnet_baseline():
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # 初始化 ResNet50 (仅微调分类头)
    model = get_resnet50_classifier(num_classes=2, feature_extract=True).to(DEVICE)
    
    # 优化器
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    print("开始单一 ResNet50 基准实验训练...")

    for epoch in range(EPOCHS):
        # 训练循环
        model.train()
        running_loss, running_acc = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_acc += torch.sum(preds == labels.data)

        # 验证循环
        model.eval()
        val_l, val_a = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_l += criterion(outputs, labels).item() * inputs.size(0)
                val_a += torch.sum(preds == labels.data)

        # 记录指标
        epoch_val_acc = val_a.double() / len(val_loader.dataset)
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['val_acc'].append(epoch_val_acc.item())
        
        print(f"Epoch {epoch} | Val Acc: {epoch_val_acc:.4f}")

        # early stopping 
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'vgg16_cats_dogs-resnet-best.pth')
            print(f"精度提升至 {best_acc:.4f}，权重已保存")
            early_stop_counter = 0 # 重置计数器
        else:
            early_stop_counter += 1
            print(f"验证集精度未提升，早停计数: {early_stop_counter}/{PATIENCE}")
            
        if early_stop_counter >= PATIENCE:
            print("触发早停，停止训练。")
            break
    plot_training_history(history)
    print(f"ResNet50 基准训练完成，最高精度: {best_acc:.4f}")

if __name__ == '__main__':
    train_resnet_baseline()