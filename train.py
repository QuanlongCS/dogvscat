import torch
import torch.nn as nn
import torch.optim as optim
from model import get_vgg16_classifier
from dataloader import get_data_loaders
from others.plot import plot_training_history
import torch.nn as nn
from model import VGG16_UNet 



NUM_CLASSES = 21 # VOC2012 包含背景共 21 类
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数
DATA_DIR = './datasets/cat_dog'
BATCH_SIZE = 128
LR = 1e-5
EPOCHS = 30
PATIENCE = 5
counter = 0
best_val_loss = float('inf') # 记录目前见过的最低验证损失
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model():
    train_loader, val_loader, _ = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # 冻结卷积层，只训练分类头,解冻最后3层卷积，调低Learning rate
    model = get_vgg16_classifier(num_classes=2, feature_extract=True)
#    model = get_vgg16_classifier(num_classes=2, feature_extract=True).to(DEVICE)
    for param in model.features[24:].parameters():param.requires_grad = True
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[6].parameters(), lr=LR)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    
    
    
    
    for epoch in range(EPOCHS):
#训练-------------------------------------------------------------
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
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
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
#损失与精确---------------------------------------------             
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f'Epoch {epoch}/{EPOCHS} Train-Loss: {train_loss:.4f} Train-Acc: {train_acc:.4f}')
        print(f'Epoch {epoch}/{EPOCHS} Val-Loss: {val_epoch_loss:.4f} Val-Acc: {val_epoch_acc:.4f}') #看的验证集结果更多
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'vgg16_cats_dogs-best.pth')
            print("模型权重已保存至 vgg16_cats_dogs-best.pth")
            print('-' * 30)
            print()
            
            
            
    # epoch 结束后绘制训练曲线
    plot_training_history(history)
        
        
        
if __name__ == '__main__':
    train_model()