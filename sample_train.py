from models.face_detector import detect_faces
from models.feature_extractor import extract_cnn_features
from models.concent_model import EngagementModel
import torch
import torch.nn as nn
import cv2

# 설정
T = 20  # 프레임 시퀀스 길이
FRAME_INTERVAL = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 레이블 가져오는 코드 작성하기 필요함
# 학습 레이블 수동 설정 (예: 집중 상태)
label = torch.tensor([0.0]).unsqueeze(0).to(device)  # (1, 1)

# 모델, 손실함수, 옵티마이저
model = EngagementModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 영상 파일 열기 (카메라 대신 avi 사용)
video_path = "./1100062016.avi"  # 테스트용 영상 하나
cap = cv2.VideoCapture(video_path)

t_faces = []
frame_count = 0

print("▶ 영상으로부터 학습용 시퀀스 수집 중...")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("⛔ 영상 끝")
        break

    if frame_count % FRAME_INTERVAL == 0:
        faces = detect_faces([frame])
        if faces:
            t_faces.append(faces[0])
            print(f"수집된 얼굴: {len(t_faces)}")

        if len(t_faces) == T:
            # 특징 추출
            feature_sequence = extract_cnn_features(t_faces, device)  # (T, 1280)
            input_seq = feature_sequence.unsqueeze(0).to(device)      # (1, T, 1280)

            # 학습 수행
            model.train()
            optimizer.zero_grad()
            output = model(input_seq)                                 # (1, 1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            print(f"✅ 1개 샘플 학습 완료 - Loss: {loss.item():.4f} - Acc: {accuracy:.2f}")

            # 학습 반복 원하면 t_faces 초기화
            t_faces.clear()

    frame_count += 1

cap.release()
