# webcam_inference_with_face_filter_binary.py

import os
import cv2
import torch
import numpy as np
from collections import deque
from torchvision import transforms
from PIL import Image

# 1) 모델 로드
from models.cnn_encoder import CNNEncoder
from models.engagement_model import EngagementModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("best_checkpoint.pth", map_location=device)

cnn = CNNEncoder().to(device)
cnn.load_state_dict(ckpt['cnn'], strict=False)
cnn.eval()

model = EngagementModel().to(device)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# 2) 전처리
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# 3) 얼굴 검출기 초기화
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def frame_to_tensor(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return preprocess(pil)

# 4) 예측 설정
LOGIT_THRESHOLD = -0.219
PROB_THRESHOLD = 1 / (1 + np.exp(-LOGIT_THRESHOLD))

def predict_buffer(buffer):
    batch = torch.stack(list(buffer), dim=0).unsqueeze(0).to(device)  # (1,30,3,224,224)
    with torch.no_grad():
        feats = cnn(batch)
        logits = model(feats).cpu().numpy().flatten()[0]
    prob = 1 / (1 + np.exp(-logits))
    pred = int(prob >= PROB_THRESHOLD)  # 0 or 1
    return pred

# 5) 실시간 캡처 루프
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    buffer = deque(maxlen=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 있는 프레임만 버퍼에 추가
        tensor = frame_to_tensor(frame)
        if tensor is not None:
            buffer.append(tensor)

        # 워밍업 중에도 pred가 정의되도록 기본값 설정
        pred = 0

        if len(buffer) == 30:
            pred = predict_buffer(buffer)

        # 화면에 결과 오버레이
        display = frame.copy()
        h, w = display.shape[:2]
        if len(buffer) < 30:
            text = f"Warming up: {len(buffer)}/30"
            color = (200, 200, 200)
        else:
            # 집중 안함(0): 빨강, 집중(1): 초록
            text = str(pred)
            color = (0,255,0) if pred == 1 else (0,0,255)

        cv2.putText(display, text, (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,
                    3, cv2.LINE_AA)

        cv2.imshow("Engagement Monitor", display)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
