import torch
import torch.nn as nn
from torchvision import models
import os



def get_vgg16_classifier(num_classes=2, feature_extract=True,use_pretrained=True):
    """
    加载预训练的VGG16并修改分类头
    feature_extract: 为True时冻结所有卷积层参数
    """

    """
    训练模式 (use_pretrained=True)：优先找本地权重，找不到就联网下载。
    推理模式 (use_pretrained=False)：直接跳过所有 ImageNet 权重加载，只初始化结构，等待加载你训练好的 .pth。
    """
    if use_pretrained:
        # 自动在线下载并加载预训练权重
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        print("VGG16: 正在在线下载/加载预训练权重...")
    else:
        model = models.vgg16(weights=None)
        print("VGG16: 推理模式，跳过预训练权重下载")
    
    #冻结参数(权重)包括全部的全连接卷积层，告诉torch不需要计算这些参数的梯度也就达到了冻结的效果
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 替换VGG16最后的Linear层
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes) #手动定义分类器
    
    return model

def get_resnet50_classifier(num_classes=2, feature_extract=True, use_pretrained=True):
    """
    加载 ResNet50 并针对猫狗分类进行微调
    feature_extract: True 表示冻结卷积层，只训练最后的分类层
    """
    # 1. 创建空架构
    model = models.resnet50(weights=None)
    
    if use_pretrained:
        # 自动在线下载 ResNet50 权重
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        print("ResNet50: 正在在线下载/加载预训练权重...")
    else:
        model = models.resnet50(weights=None)
        print("ResNet50: 推理模式，跳过预训练权重下载")

    # 3. 冻结参数逻辑 (迁移学习的核心)
    # 
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False #停止梯度更新 = 冻结

    # 4. 替换 ResNet50 的全连接层 (fc)
    # ResNet50 最后一层默认输出 1000 维（ImageNet 类别）
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model



class VGG_ResNet_Fusion(nn.Module):
    def __init__(self, num_classes=2, vgg_path=None, resnet_path=None):
        super(VGG_ResNet_Fusion, self).__init__()
        
        # 1. 初始化基础骨架
        vgg = models.vgg16(weights=None)
        resnet = models.resnet50(weights=None)
        
        # 2. 载入你之前训练好的 VGG 专家权重 (只取 features 部分)
        if vgg_path and os.path.exists(vgg_path):
            state_dict = torch.load(vgg_path, map_location='cpu')
            # 过滤掉原有的分类头
            feat_dict = {k.replace('features.', ''): v for k, v in state_dict.items() if 'features' in k}
            vgg.features.load_state_dict(feat_dict)
            print(f"✅ Fusion: 已装载 VGG 专家特征层")
        else:
            # 如果没提供专家权重，则在线下载 ImageNet 权重作为保底
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # 3. 载入你之前训练好的 ResNet 专家权重 (只取 backbone 部分)
        if resnet_path and os.path.exists(resnet_path):
            state_dict = torch.load(resnet_path, map_location='cpu')
            # 过滤掉原有的 fc 层
            feat_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
            resnet.load_state_dict(feat_dict, strict=False)
            print(f"✅ Fusion: 已装载 ResNet50 专家特征层")
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 提取特征层
        self.vgg_features = vgg.features
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 4. 增强型融合分类头 (这是冲刺高分的关键)
        self.classifier = nn.Sequential(
            nn.Linear(2560, 1024),
            nn.BatchNorm1d(1024), # 特征对齐：让 VGG 和 ResNet 的声音一样大
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f_vgg = self.gap(self.vgg_features(x)).view(x.size(0), -1)
        f_resnet = self.gap(self.resnet_features(x)).view(x.size(0), -1)
        # 拼接维度: 512 + 2048 = 2560
        combined = torch.cat((f_vgg, f_resnet), dim=1)
        return self.classifier(combined)



if __name__ == '__main__':
    # 测试模型构建
    model = get_vgg16_classifier(num_classes=2, feature_extract=True)
    # model = get_fusion_model(num_classes=2, feature_extract=True)
    # model = VGG_ResNet_Fusion(num_classes=2)
    print(model)