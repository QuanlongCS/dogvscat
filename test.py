import torch
from model import get_vgg16_classifier
from dataloader import get_data_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test_folder():
    # 1. 加载模型
    model = get_vgg16_classifier(num_classes=2)
    model.load_state_dict(torch.load('vgg16_cats_dogs.pth'))
    model = model.to(DEVICE)
    model.eval()

    # 2. 加载无标签测试集
    _, test_loader = get_data_loaders('./datasets/cat_dog')

    classes = ['Cat', 'Dog']
    print("开始推理测试集...")

    with torch.no_grad():
        for inputs, img_names in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(len(img_names)):
                print(f"文件: {img_names[i]} -> 预测结果: {classes[preds[i].item()]}")


if __name__ == '__main__':
    predict_test_folder()