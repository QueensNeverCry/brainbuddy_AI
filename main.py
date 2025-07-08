from detection.frame_extractor import extract_frames
from detection.face_detector import detect_faces
from detection.sequence_builder import build_sequences
from models.feature_extractor import extract_cnn_features
from models.concent_model import SimpleLSTM
from utils.visualization import show_image
import time
import torch
import torch.nn as nn
import cv2

T = 10  # 시퀀스 길이
FRAME_INTERVAL = 6  # 실시간: 6프레임마다 1장 수집(그래서 1초동안 5장 수집)
# 그래서 약 2초 분량의 영상을 한번에 분석하는 구조

model = SimpleLSTM()
loss_fn = nn.BCELoss()
label = torch.tensor([0.0]).unsqueeze(1)

cap = cv2.VideoCapture(0)
all_faces = []
count = 0
frames_without_face = 0  # 얼굴을 인식하지 못한 프레임 수

FPS = 30
NO_FACE_THRESHOLD = FPS * 10  # 10초 (예: 30fps × 10 = 300)
OUTPUT_INTERVAL = FPS * 2     # 2초마다 출력

print("실시간 영상 스트리밍 시작 (q: 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 일정 프레임 간격마다 처리
    if count % FRAME_INTERVAL == 0:#0.2초에 한번 처리
        #얼굴 검출
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces([rgb_frame])
        #얼굴 탐지
        if faces:
            all_faces.append(faces[0])
            print(f"수집된 얼굴: {len(all_faces)}")

            # 시퀀스가 T개 모이면 모델에 넣어 예측
            if len(all_faces) >= T:
                sequence = all_faces[-T:]
                feature_sequence = extract_cnn_features(sequence)
                output = model(feature_sequence)
                print(f"▶ 집중도: {output.item():.4f}")
        else:
            frames_without_face += FRAME_INTERVAL
            # 10초 이상 얼굴 인식 실패 시 2초마다 출력
            if frames_without_face >= NO_FACE_THRESHOLD and frames_without_face % OUTPUT_INTERVAL == 0:
                print("❗ 얼굴 미검출 상태 지속 중... ▶ 집중도: 0.0000")
    
    count += 1

    cv2.imshow("Live", frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):#q키를 누르면 종료되도록
        break

cap.release()
cv2.destroyAllWindows()
