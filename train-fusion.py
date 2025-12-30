import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from models.model import VGG_ResNet_Fusion
from others.dataloader import get_data_loaders
from others.plot import plot_training_history


RES_BEST = '.models/best-model/best-resnet-0.9934.pth'
VGG_BEST = '.models/best-model/best-vgg-0.9916.pth'


# --- 超参数配置 ---
DATA_DIR = '/public/home/liuquanlong_gsc/Datasets/Dogvscat/'
BATCH_SIZE = 32  # 融合模型较大，建议调小 Batch 避免 OOM
EPOCHS = 50
LR = 1e-4   # 分类头的学习率
LR_HEAD = 1e-4
LR_BACKBONE = 1e-6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_fusion():
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)
    model = VGG_ResNet_Fusion(num_classes=2, vgg_path=VGG_BEST, resnet_path=RES_BEST).to(DEVICE)
    # 1. 初始化融合模型

    optimizer = optim.Adam([
        {'params': model.vgg_features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.resnet_features.parameters(), 'lr': LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD}
    ])
    
        
        
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels,_ in train_loader:
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
            for inputs, labels,_ in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        # 计算指标
        train_loss = running_loss / len(train_loader.dataset)
        #train_a = running_corrects.double() / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch}/{EPOCHS} Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        
        history['train_loss'].append(train_loss)
        #history['train_acc'].append(train_acc.item())
        #history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        scheduler.step(val_loss)
        
        
        # --- 最优保存与早停 ---
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'./pths/best-fusion-{best_acc:.4f}.pth')
            print(f"精度突破！已保存: {best_acc:.4f}")


        

    plot_training_history(history,"fusion_training_curves")

if __name__ == '__main__':
    train_fusion()