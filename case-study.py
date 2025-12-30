import torch
import torch.nn.functional as F
import os
import shutil
from others.dataloader import get_data_loaders
from models.model import get_resnet50_classifier, VGG_ResNet_Fusion

# --- 配置路径与模型 ---
DATA_DIR = '/public/home/liuquanlong_gsc/Datasets/Dogvscat/'
RESNET_PATH = './models/best-model/best-resnet-0.9934.pth'
FUSION_PATH = './models/best-model/best-fusion-0.9944.pth'
VGG_PATH = './models/best-model/best-vgg-0.9916.pth'
SAVE_DIR = './case_study_results'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_case_study():
    # 创建结果目录
    for folder in ['fusion_win', 'resnet_win', 'both_wrong', 'fusion_confident_error']:
        os.makedirs(os.path.join(SAVE_DIR, folder), exist_ok=True)

    # 1. 加载模型
    resnet = get_resnet50_classifier(num_classes=2, use_pretrained=False).to(DEVICE)
    resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    resnet.eval()

    fusion = VGG_ResNet_Fusion(num_classes=2).to(DEVICE)
    fusion.load_state_dict(torch.load(FUSION_PATH, map_location=DEVICE))
    fusion.eval()

    # 2. 使用验证集 (因为有 Ground Truth 才能判断对错)
    _, val_loader, _ = get_data_loaders(DATA_DIR, batch_size=1)

    print("开始扫描差异样本...")
    
    with torch.no_grad():
        for inputs, labels, img_names in val_loader: # 注意 dataloader 需返回 img_names
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # ResNet 预测
            out_r = resnet(inputs)
            prob_r = F.softmax(out_r, dim=1)
            val_r, pred_r = torch.max(prob_r, 1)

            # Fusion 预测
            out_f = fusion(inputs)
            prob_f = F.softmax(out_f, dim=1)
            val_f, pred_f = torch.max(prob_f, 1)

            img_name = img_names[0]
            label = labels.item()
            pr, pf = pred_r.item(), pred_f.item()
            
            # 找到原图路径 (假设在 val 目录下)
            src_path = os.path.join(DATA_DIR, 'train', img_name)

            # --- 分类逻辑 ---
            # 1. Fusion 赢了 (ResNet 错, Fusion 对)
            if pr != label and pf == label:
                shutil.copy(src_path, os.path.join(SAVE_DIR, 'fusion_win', img_name))
            
            # 2. ResNet 赢了 (Fusion 错, ResNet 对)
            elif pr == label and pf != label:
                shutil.copy(src_path, os.path.join(SAVE_DIR, 'resnet_win', img_name))

            # 3. 两个都错了
            elif pr != label and pf != label:
                shutil.copy(src_path, os.path.join(SAVE_DIR, 'both_wrong', img_name))
            
            # 4. 导致 Log Loss 爆炸的“过自信错误”
            if pf != label and val_f > 0.95:
                shutil.copy(src_path, os.path.join(SAVE_DIR, 'fusion_confident_error', img_name))

    print(f"扫描完成！结果保存在 {SAVE_DIR}")

if __name__ == '__main__':
    run_case_study()