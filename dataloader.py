import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', size=(224, 224)):
        self.root_dir = root_dir
        self.size = size
        # 根据 train.txt 或 val.txt 读取文件名
        txt_path = os.path.join(root_dir, 'ImageSets/Segmentation', f'{split}.txt')
        with open(txt_path, 'r') as f:
            self.images = [line.strip() for line in f]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, 'JPEGImages', img_name + '.jpg')
        mask_path = os.path.join(self.root_dir, 'SegmentationClass', img_name + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path) # 保持索引模式

        # --- 同步数据增强 ---
        # 1. 统一缩放
        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # 2. 转换为 Tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask 转换为 LongTensor，且像素值 255 保持不变供 Loss 忽略
        mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask

def get_voc_loaders(data_dir, batch_size=32):
    train_ds = VOCSegmentationDataset(data_dir, split='train')
    val_ds = VOCSegmentationDataset(data_dir, split='val')
    
    # 针对你的 4090 服务器，建议加大 num_workers
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)
    
    return train_loader, val_loader