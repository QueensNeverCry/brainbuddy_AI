from detection.face_detector import detect_faces
from models.feature_extractor import extract_cnn_features
from models.concent_model import SimpleLSTM
from utils.visualization import show_image
import time
import torch
import torch.nn as nn
import cv2

T = 20  # 시퀀스 길이
FRAME_INTERVAL = 6  # 실시간: 6프레임마다 1장 수집(그래서 1초동안 5장 수집)
# 그래서 약 4초 분량의 영상을 한번에 분석하는 구조
FPS = 30
NO_FACE_THRESHOLD = FPS * 10
OUTPUT_INTERVAL = FPS * 2 

model = SimpleLSTM()
loss_fn = nn.BCELoss()
label = torch.tensor([0.0]).unsqueeze(1)
cap = cv2.VideoCapture(0) # 0번 카메라 열기. 기본 카메라

t_faces = []
count = 0
frames_without_face = 0  # 얼굴을 인식하지 못한 프레임 수

print("실시간 영상 스트리밍 시작 (q: 종료)")

if cap.isOpened():
    while True:
        ret, frame = cap.read() # ret: T/F
        if not ret: #새로운 프레임이 없는 경우 종료
            break

        # 일정 프레임 간격마다 처리
        if count % FRAME_INTERVAL == 0: # 0.2초에 한번 처리
            #얼굴 검출           
            faces = detect_faces([frame]) # rgb_frame: (높이, 너비, 3)을 하나의 원소로 갖는 리스트로 전달
            #얼굴 탐지
            if faces:
                t_faces.append(faces[0])# 한 프레임에서 여러 얼굴이 검출되어도 한 얼굴만 사용
                print(f"수집된 얼굴: {len(t_faces)}")

                # 시퀀스가 T개 모이면 모델에 넣어 예측
                if len(t_faces) >= T:
                    feature_sequence = extract_cnn_features(t_faces)
                    output = model(feature_sequence)
                    print(f"▶ 집중도: {output.item():.4f}")
                    
                    t_faces.clear()# 리스트 초기화
            else:
                frames_without_face += FRAME_INTERVAL #얼굴이 검출되지 않을 때마다 증가
                if frames_without_face >= NO_FACE_THRESHOLD and frames_without_face % OUTPUT_INTERVAL == 0:
                    print("❗ 얼굴 미검출 상태 지속 중... ▶ 집중도: 0.0000")     
        count += 1

        cv2.imshow("Live", frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):#q키를 누르면 종료되도록
            break
else :
    print("카메라가 성공적으로 열리지 않았습니다.")
cap.release()
cv2.destroyAllWindows()
