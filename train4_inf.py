import os
import torch
from PIL import Image
from torchvision import transforms
from models.cnn_encoder import CNNEncoder
from models.engagement_model import EngagementModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 동일한 transform 사용
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_video_tensor_from_folder(folder_path, transform):
    img_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if len(img_files) < 30:
        raise ValueError(f"❌ 이미지가 30장 미만입니다: {folder_path}")

    selected_files = img_files[:30]
    frames = []
    for fname in selected_files:
        img_path = os.path.join(folder_path, fname)
        image = Image.open(img_path).convert("RGB")
        if transform:
            image = transform(image)
        frames.append(image)

    video = torch.stack(frames)  # (30, 3, 224, 224)
    return video.unsqueeze(0)    # (1, 30, 3, 224, 224)

def run_inference(cnn_path, lstm_path, video_folder_path):
    # ✅ 모델 불러오기
    cnn = CNNEncoder().to(device)
    lstm = EngagementModel().to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    lstm.load_state_dict(torch.load(lstm_path, map_location=device))
    cnn.eval()
    lstm.eval()

    # ✅ 데이터 로드
    video_tensor = load_video_tensor_from_folder(video_folder_path, transform).to(device)

    # ✅ 추론
    with torch.no_grad():
        features = cnn(video_tensor)         # (1, 30, feature_dim)
        logits = lstm(features)              # (1, 1)
        prob = torch.sigmoid(logits).item()  # 확률로 변환
        pred = int(prob >= 0.5)

    print(f"📁 Folder: {video_folder_path}")
    print(f"🧠 예측 확률: {prob:.4f} → 예측 클래스: {pred}")
    return prob, pred


if __name__ == "__main__":
    fold_num = 3  # 예시로 Fold 3 모델 사용
    video_folder = "C:/KSEB/brainbuddy_AI/some_test_sample"

    cnn_model_path = f"best_cnn_fold{fold_num}.pth"
    lstm_model_path = f"best_lstm_fold{fold_num}.pth"

    run_inference(cnn_model_path, lstm_model_path, video_folder)

# 여러 폴더에 대해 반복추론을 원하는경우
# test_root = "C:/KSEB/brainbuddy_AI/test_set"

# for folder in os.listdir(test_root):
#     folder_path = os.path.join(test_root, folder)
#     if os.path.isdir(folder_path):
#         run_inference(cnn_model_path, lstm_model_path, folder_path)
