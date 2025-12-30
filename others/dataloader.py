#我们将测试集逻辑加入，并增加一个专门用于单张图片推理的转换（无需随机增强）。
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from natsort import natsorted
import csv

class CatDogDataset(Dataset):
    def __init__(self, root_dir, image_files,transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        # 列表推导式, 读取目录下所有图片,把所有以 .jpg 结尾的文件名挑出来，组成一个新的列表
        #self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        #self.images = sorted(image_files)  # 这里传进来的是图片文件名列表
        #sorted 这里保证test输出的顺序性，可能导致读取的慢记得改回来
        
        self.images = natsorted(image_files)
        
        
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # test 文件夹没有标签，直接返回文件名
            return image, img_name
        else:
            # train 文件夹解析文件名获取标签
            # cat.0.jpg -> 0 (Cat), dog.0.jpg -> 1 (Dog)
            label = 1 if 'dog' in img_name.lower() else 0
            return image, label, img_name


def get_data_loaders(data_dir, batch_size=64,val_split=0.2):
    norm_params = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    # 优化：增加随机旋转和色彩亮度调节
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15), # 随机旋转 ±15 度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 色彩抖动
        transforms.ToTensor(),
        transforms.Normalize(**norm_params)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**norm_params)
    ])
    
    
    # 1. 获取 train 目录下所有图片并打乱
    train_path = os.path.join(data_dir, 'train')
    all_train_files = [f for f in os.listdir(train_path) if f.endswith('.jpg')]
    random.seed(42) # 固定随机种子保证结果可复现
    random.shuffle(all_train_files)
    # 2. 计算切分点
    split = int(len(all_train_files) * (1 - val_split))
    train_files = all_train_files[:split]
    val_files = all_train_files[split:]
    # 3. 创建带标签的训练集和验证集
    train_dataset = CatDogDataset(train_path, train_files, transform=train_transform, is_test=False)
    val_dataset = CatDogDataset(train_path, val_files, transform=val_transform, is_test=False)
    # 4. 创建无标签的测试集
    test_path = os.path.join(data_dir, 'test')
    test_files = [f for f in os.listdir(test_path) if f.endswith('.jpg')]
    test_dataset = CatDogDataset(test_path, test_files, transform=val_transform, is_test=True)

    # 返回三个 Loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader