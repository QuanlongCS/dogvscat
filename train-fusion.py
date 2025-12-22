import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.model_fusion import VGG_ResNet_Fusion
from dataloader import get_data_loaders
from others.plot import plot_training_history

# --- 超参数配置 ---
DATA_DIR = '/public/home/liuquanlong_gsc/Datasets/Dogvscat/'
BATCH_SIZE = 64  # 融合模型较大，建议调小 Batch 避免 OOM
EPOCHS = 100
LR_HEAD = 1e-4   # 分类头的学习率
LR_BACKBONE = 1e-6 # 预训练层的微调学习率
PATIENCE = 7     # 早停耐心值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fusion():
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # 1. 初始化融合模型
    model = VGG_ResNet_Fusion(num_classes=2).to(DEVICE)
    
    # 2. 差异化微调策略：解冻 VGG 的最后 3 层和 ResNet 的最后 2 个残差块
    for param in model.vgg_features[24:].parameters(): param.requires_grad = True
    for param in list(model.resnet_features.parameters())[-20:]: param.requires_grad = True

    # 3. 优化器：为不同层设置不同的学习率（冲 99% 的关键）
    optimizer = optim.Adam([
        {'params': model.vgg_features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.resnet_features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ])

    # 4. 动态学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 增加标签平滑防止过拟合
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # --- 验证阶段 ---
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        # 计算指标
        train_l = running_loss / len(train_loader.dataset)
        #train_a = running_corrects.double() / len(train_loader.dataset)
        val_l = val_loss / len(val_loader.dataset)
        val_a = val_corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch}/{EPOCHS} Train Loss: {train_l:.4f} | Val Acc: {val_a:.4f}')
        
        scheduler.step(val_l) # 更新学习率
        history['train_loss'].append(train_l)
        #history['train_acc'].append(train_a.item())
        #history['val_loss'].append(val_l)
        history['val_acc'].append(val_a.item())

        # --- 最优保存与早停 ---
        if val_a > best_acc:
            best_acc = val_a
            torch.save(model.state_dict(), 'fusion_best_99.pth')
            print(f"精度突破！已保存: {best_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= PATIENCE:
            print("模型已收敛，触发早停。")
            break
    plot_training_history(history)

if __name__ == '__main__':
    train_fusion()