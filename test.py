import torch
import torch.nn.functional as F
import csv
import os
from natsort import natsorted # 推荐使用自然排序
from others.dataloader import get_data_loaders

from models.model import get_vgg16_classifier
from models.model import get_resnet50_classifier
from models.model import VGG_ResNet_Fusion
"""
resnet 0.9890
vgg16 0.9908
fusion 0.9926


"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_to_csv(model_type='vgg'):
    # 1. 加载模型

    if model_type == 'vgg':
        model = get_vgg16_classifier(num_classes=2, use_pretrained=False)
        weight_path = './models/best-model/best-vgg-0.9916.pth'
    elif model_type == 'resnet':
        model = get_resnet50_classifier(num_classes=2, use_pretrained=False)
        weight_path = './models/best-model/best-resnet-0.9934.pth'
    elif model_type == 'fusion':
        model = VGG_ResNet_Fusion(num_classes=2, vgg_path='./models/best-model/best-vgg-0.9916.pth', resnet_path='./models/best-model/best-resnet-0.9934.pth')
        weight_path = './models/best-model/best-fusion-0.9944.pth'
    
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 2. 加载测试集 (确保 shuffle=False)
    _, _, test_loader = get_data_loaders('/public/home/liuquanlong_gsc/Datasets/Dogvscat/', batch_size=1)

    results = []
    print("开始推理并计算概率值...")

    with torch.no_grad():
        for inputs, img_names in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs) # 得到的是原始 Logits
            
            # --- 关键修改：计算概率 ---
            # 使用 Softmax 将 Logits 转换为概率
            # 公式: $$P(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}$$
            probabilities = F.softmax(outputs, dim=1)
            dog_probs = probabilities[:, 1]
            # 获取预测类别的概率值和索引
            #prob_values, preds = torch.max(probabilities, 1)

            for i in range(len(img_names)):
                filename = img_names[i]
            # 这里的 filename 可能是 '123.jpg'，Kaggle 有时只需要 '123'
            # 建议根据 Kaggle 要求处理文件名
                img_id = filename.split('.')[0] 
            
                conf_score = dog_probs[i].item()
                results.append([img_id, f"{conf_score:.4f}"])
    # 3. 排序 (解决 1, 10, 100 顺序问题)
    results = natsorted(results, key=lambda x: x[0])

    # 4. 写入 CSV
    if model_type == 'vgg':
        csv_path = 'classification_probs-vgg.csv'
    elif model_type == 'resnet':
        csv_path = 'classification_probs-resnet.csv'
    else:
        csv_path = 'classification_probs-fusion.csv'
        
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label']) # label 处现在是概率值
        writer.writerows(results)

    print(f"推理完成！结果已保存至: {csv_path}")

if __name__ == '__main__':
    predict_to_csv(model_type='vgg')  # 可选 'vgg', 'resnet', 'fusion'