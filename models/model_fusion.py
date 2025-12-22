import torch
import torch.nn as nn
from torchvision import models

class VGG_ResNet_Fusion(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VGG_ResNet_Fusion, self).__init__()
        
        # 1. 加载 VGG16 特征提取部分、加载 ResNet50 并去掉最后的全连接层 
        vgg = models.vgg16(pretrained=pretrained)
        self.vgg_features = vgg.features # 输出通道 512
        resnet = models.resnet50(pretrained=pretrained)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2]) # 输出通道 2048
        
        # 3. 全局平均池化 (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 融合分类头
        # 拼接后的维度: 512 (VGG) + 2048 (ResNet) = 2560
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # 防止融合模型过拟合
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # VGG 分支特征提取
        x_vgg = self.vgg_features(x)
        x_vgg = self.gap(x_vgg).view(x_vgg.size(0), -1) # [Batch, 512]
        
        # ResNet 分支特征提取
        x_resnet = self.resnet_features(x)
        x_resnet = self.gap(x_resnet).view(x_resnet.size(0), -1) # [Batch, 2048]
        
        # 特征拼接 (Feature Fusion)
        combined = torch.cat((x_vgg, x_resnet), dim=1) # [Batch, 2560]
        
        # 分类输出
        out = self.classifier(combined)
        return out

def get_fusion_model(num_classes=2):
    return VGG_ResNet_Fusion(num_classes=num_classes)