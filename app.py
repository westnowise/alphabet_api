from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# 데이터 변환 정의 (그레이스케일)
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 그레이스케일로 변환
    transforms.Resize((224, 224)),  # 이미지 크기를 224x224로 조정
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 그레이스케일 이미지의 평균 및 표준편차로 정규화
])

# 클래스 이름 정의
class_names = [chr(i) for i in range(97, 123)] + [chr(i) for i in range(65, 91)]

# ResNet 모델 불러오기 및 수정 (ResNet-18)
model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 첫 번째 레이어를 그레이스케일 입력으로 수정
model_ft.fc = torch.nn.Linear(num_ftrs, len(class_names))

# 학습된 모델의 가중치 불러오기
model_path = 'alphabet_resnet18_2.pth'
model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model_ft.eval()

# 이미지 예측 함수 정의
def predict_image(image, model, class_names):
    image = data_transforms(image)
    image = image.unsqueeze(0)  # Batch size of 1

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    return label

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('L')  # 'L' 모드를 사용해 그레이스케일로 변환

    label = predict_image(image, model_ft, class_names)
    return jsonify({'predicted_label': label})

if __name__ == '__main__':
    app.run(debug=True)
