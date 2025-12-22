import torch
import torch.nn as nn
from torchvision import models
import os

def get_vgg16_classifier(num_classes=2, feature_extract=True):
    """
    加载预训练的VGG16并修改分类头
    feature_extract: 为True时冻结所有卷积层参数
    """
    model = models.vgg16(weights=None)
    weights_path = "./models/vgg16-397923af.pth"
    
    # 加载本地预训练权重
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)
        print(f"本地加载预训练权重: {weights_path}")
    else:
        print(f"⚠️ 本地权重文件不存在: {weights_path},下载预训练权重...")
        model = models.vgg16(pretrained=True)
    
    
    #冻结参数(权重)包括全部的全连接卷积层，告诉torch不需要计算这些参数的梯度也就达到了冻结的效果
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 替换VGG16最后的Linear层
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes) #手动定义分类器
    
    return model


if __name__ == '__main__':
    # 测试模型构建
    model = get_vgg16_classifier(num_classes=2, feature_extract=True)
    print(model)