import torch
import torch.nn as nn
from torchvision import models
import os

import torch
import torch.nn as nn
from torchvision import models

class VGG16_UNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(VGG16_UNet, self).__init__()
        
        # 1. 借用 VGG16 的 features 部分 自动下载到cache目录
        vgg = models.vgg16(pretrained=pretrained).features
        
        """
        # model = models.vgg16(weights=None)
        # weights_path = "./others/vgg16-397923af.pth"
    
    # 加载本地预训练权重
        # if os.path.exists(weights_path):
        #     state_dict = torch.load(weights_path)
        #     model.load_state_dict(state_dict)
        #     print(f"本地加载预训练权重: {weights_path}")
        # else:
        #     print(f"⚠️ 本地权重文件不存在: {weights_path},下载预训练权重...")
        #     model = models.vgg16(pretrained=True)
        """
        
        
        # 2. 拆解 VGG 块，提取 5 个层级的特征用于 Skip Connections
        self.block1 = vgg[:4]    # 输出: 64通道 (与输入同尺寸)
        self.block2 = vgg[4:9]   # 输出: 128通道 (1/2 尺寸)
        self.block3 = vgg[9:16]  # 输出: 256通道 (1/4 尺寸)
        self.block4 = vgg[16:23] # 输出: 512通道 (1/8 尺寸)
        self.block5 = vgg[23:30] # 输出: 512通道 (1/16 尺寸, 瓶颈层)

        # 3. 手写 Decoder (上采样)
        self.up4 = self.decoder_block(512, 256) # 处理 block4 传来的特征
        self.up3 = self.decoder_block(256 + 512, 128) # +512 是因为 Concatenate
        self.up2 = self.decoder_block(128 + 256, 64)
        self.up1 = self.decoder_block(64 + 128, 32)
        
        # 最后的输出层
        self.final = nn.Conv2d(32 + 64, num_classes, kernel_size=1)

    def decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder 路径 + 保存特征图用于 Skip
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        b  = self.block5(s4) # 瓶颈层特征

        # Decoder 路径 + Concatenate (拼接)
        x = self.up4(b)
        x = torch.cat([x, s4], dim=1)
        
        x = self.up3(x)
        x = torch.cat([x, s3], dim=1)
        
        x = self.up2(x)
        x = torch.cat([x, s2], dim=1)
        
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)

        return self.final(x)

if __name__ == '__main__':
    # 测试模型构建
    model = VGG16_UNet()
    print(model)