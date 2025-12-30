import torch
import torch.nn as nn
import torch.optim as optim
from models.model import get_vgg16_classifier
from others.dataloader import get_data_loaders
from others.plot import plot_training_history
import torch.nn as nn
from models.model import get_vgg16_classifier 
from torch.optim import lr_scheduler # 引入调度器



# 超参数
DATA_DIR = '/public/home/liuquanlong_gsc/Datasets/Dogvscat/'
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4

PATIENCE = 7
counter = 0
best_val_loss = float('inf') # 记录目前见过的最低验证损失
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model():
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # 冻结卷积层，只训练分类头,解冻最后3层卷积，调低Learning rate
    model = get_vgg16_classifier(num_classes=2, feature_extract=True)
    
    
    for param in model.features[24:].parameters():
        param.requires_grad = True
    model = model.to(DEVICE)
    
# 1. 动态获取所有需要更新（未冻结）的参数
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=LR)
    #check out which parameters are being optimized
    # 健壮性检查：确保有参数可练
    if len(params_to_update) == 0:
        raise ValueError("错误：没有任何参数被解冻！请检查你的 feature_extract 逻辑。")
    # 2. 修复优化器# 注意：确保 LR 已经在脚本全局定义，例如 LR = 1e-5

    # 3. 修复调度器
    # 移除了 verbose=True 以兼容最新版 PyTorch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # 4. 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    early_stop_counter = 0 # 初始化早停计数器
    
    
    
    for epoch in range(EPOCHS):
#训练-------------------------------------------------------------
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels,_ in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
#验证--------------------------------------------
        model.eval() 
        val_loss, val_corrects = 0.0, 0
        
        with torch.no_grad(): # 验证时不计算梯度
            for inputs, labels,_ in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
#损失与精确---------------------------------------------      

        train_loss = running_loss / len(train_loader.dataset)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)        
        #train_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Epoch {epoch}/{EPOCHS} Train-Loss: {train_loss:.4f} Val-Acc: {val_epoch_acc:.4f}')
        #print(f'Epoch {epoch}/{EPOCHS} Val-Loss: {val_epoch_loss:.4f} Val-Acc: {val_epoch_acc:.4f}') #看的验证集结果更多
        
        history['train_loss'].append(train_loss)
        #history['train_acc'].append(train_acc.item())
        #history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        
        scheduler.step(val_epoch_loss) # 根据验证损失调整学习率
        
        
        # early stopping 
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), f'./pths/best-vgg-{best_acc:.4f}.pth')
            print(f"精度提升至 {best_acc:.4f}，权重已保存")
            early_stop_counter = 0 # 重置计数器
        else:
            early_stop_counter += 1
            print(f"验证集精度未提升，早停计数: {early_stop_counter}/{PATIENCE}")
            
        if early_stop_counter >= PATIENCE:
            print("触发早停，停止训练。")
            break
            
            
    # epoch 结束后绘制训练曲线
    plot_training_history(history, 'vgg16_finetune_training_curves')
    
        
        
        
if __name__ == '__main__':
    train_model()