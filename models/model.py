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
        weights_path = "./vgg16-397923af.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
            print(f"加载 ImageNet 预训练权重成功")
        else:
            print(f"⚠️ 本地权重文件不存在: {weights_path},下载预训练权重...")
            model = models.vgg16(pretrained=True)
    else:
        # 逻辑 C：推理阶段，不需要加载 ImageNet 权重
        model = models.vgg16(weights=None)
        print("推理模式：跳过 ImageNet 权重加载，等待载入自定义模型...")
    
    #冻结参数(权重)包括全部的全连接卷积层，告诉torch不需要计算这些参数的梯度也就达到了冻结的效果
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 替换VGG16最后的Linear层
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes) #手动定义分类器
    
    return model


def get_resnet50_classifier(num_classes=2,feature_extract=True,use_pretrained=True):
    if use_pretrained:
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            print("已通过 torchvision API 加载 ResNet50 预训练权重")
        except ImportError:
            model = models.resnet50(pretrained=True)
            print("已通过旧版 API 加载 ResNet50 预训练权重")
    else:
        model = models.resnet50(weights=None)

    # 2. 冻结参数逻辑
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    # 3. 替换 ResNet50 最后的 FC 层
    # ResNet50 的最后一层叫 fc (VGG 叫 classifier)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


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
def get_fusion_model(num_classes=2, pretrained=True):
    return VGG_ResNet_Fusion(num_classes=num_classes, pretrained=pretrained)
if __name__ == '__main__':
    # 测试模型构建
    model = get_vgg16_classifier(num_classes=2, feature_extract=True)
    # model = get_fusion_model(num_classes=2, feature_extract=True)
    # model = VGG_ResNet_Fusion(num_classes=2)
    print(model)