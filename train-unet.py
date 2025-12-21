import torch.nn as nn
from model import VGG16_UNet # 使用我们刚才写的 UNet 骨架
from dataloader import get_voc_loaders
import torch

# 4090 服务器配置
BATCH_SIZE = 64 
NUM_CLASSES = 21 # VOC2012 包含背景共 21 类
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_unet():

    train_loader, val_loader = get_voc_loaders('./VOCdevkit/VOC2012', BATCH_SIZE)
    
    # 初始化我们预训练好的 VGG-UNet
    model = VGG16_UNet(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 关键点：忽略 255 边缘像素
    criterion = nn.CrossEntropyLoss(ignore_index=255) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(30): # 分割任务收敛较慢，建议多跑几轮
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images) # 输出维度: [Batch, 21, 224, 224]
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
        # 每隔几个 Epoch 保存一次模型用于查看分割效果
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f'unet_voc_epoch{epoch}.pth')
            
            
            
import matplotlib.pyplot as plt
import numpy as np
import torch

def decode_segmap(image, nc=21):
    """将索引图转换为标准的 VOC 彩色图"""
    label_colors = np.array([(0, 0, 0),  # 0=background
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def visualize_result(model, dataset, device, index=0):
    model.eval()
    image, mask = dataset[index]
    # 增加 batch 维度并发送到显卡
    input_tensor = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # 寻找概率最大的类别
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 处理原图显示
    img_show = image.permute(1, 2, 0).numpy()
    img_show = img_show * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406] # 反归一化
    img_show = np.clip(img_show, 0, 1)

    # 绘图
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_show)
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(decode_segmap(mask.numpy()))
    plt.title("Ground Truth")
    
    plt.subplot(1, 3, 3)
    plt.imshow(decode_segmap(pred))
    plt.title("VGG-UNet Prediction")
    
    plt.show()
    plt.savefig(f'result_{index}.png')
    
    
    
    


