import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.model import get_resnet50_classifier # 确保你已创建此模型文件
from others.dataloader import get_data_loaders
from others.plot import plot_training_history

# --- 实验配置（保持与 VGG 实验一致以保证公平对比） ---
DATA_DIR = '/public/home/liuquanlong_gsc/Datasets/Dogvscat/'
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 15
counter = 0
best_val_loss = float('inf') # 记录目前见过的最低验证损失



def train_resnet_baseline():
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # 初始化 ResNet50 (仅微调分类头)
    model = get_resnet50_classifier(num_classes=2, feature_extract=True).to(DEVICE)
    
    # 解冻 ResNet50 的第四个大块（Layer 4），让它学习猫狗的特有高层语义；如果你想让 ResNet 效果超越 VGG，可以放开最后几层
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=LR)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 在训练代码中修改
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

    print("开始单一 ResNet50 基准实验训练...")

    for epoch in range(EPOCHS):
        # 训练循环
        model.train()
        running_loss, running_acc = 0.0, 0
        for inputs, labels,_ in train_loader:
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
            for inputs, labels,_ in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_l += criterion(outputs, labels).item() * inputs.size(0)
                val_a += torch.sum(preds == labels.data)

        # 记录指标
        epoch_val_acc = val_a.double() / len(val_loader.dataset)
        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['val_acc'].append(epoch_val_acc.item())
        
        #print(f"Epoch {epoch} | Val Acc: {epoch_val_acc:.4f}")
        print(f'Epoch {epoch}/{EPOCHS} Train-Loss: {running_loss / len(train_loader.dataset):.4f} Val-Acc: {epoch_val_acc:.4f}')
        # early stopping 
        
        scheduler.step(val_l / len(val_loader.dataset))
        
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), f'./pths/best-resnet-{best_acc:.4f}.pth')
            print(f"精度提升至 {best_acc:.4f}，权重已保存")
            early_stop_counter = 0 # 重置计数器
        else:
            early_stop_counter += 1
            print(f"验证集精度未提升，早停计数: {early_stop_counter}/{PATIENCE}")
            
        if early_stop_counter >= PATIENCE:
            print("触发早停，停止训练。")
            break
    plot_training_history(history,"resnet50_baseline_training_curves")
    print(f"ResNet50 基准训练完成，最高精度: {best_acc:.4f}")

if __name__ == '__main__':
    train_resnet_baseline()