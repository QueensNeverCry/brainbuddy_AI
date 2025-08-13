# real_time_5sec_stride.py
import time
import cv2
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from PIL import Image
import mediapipe as mp

from models.face_crop2 import crop_face  # (face_rgb, bbox) 반환

# ===== 설정 =====
CKPT_PATH = "./log/train4/best_model/best_model_epoch_4.pt"  # <-- 실제 경로로 변경
CAM_INDEX = 0
TARGET_FPS = 3
IMG_SIZE = 224
T_WINDOW = 15                                 # 모델 입력 길이(프레임) ≈ 10초
STRIDE_SEC = 3                                # 5초마다 결과 갱신
STRIDE_FRAMES = max(1, int(round(TARGET_FPS * STRIDE_SEC)))  # 3*5=15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 전처리 =====
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ===== 모델 =====
class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512, dropout2d=0.1, proj_dropout=0.4):
        super().__init__()
        w = models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = models.mobilenet_v3_large(weights=w)
        self.features = backbone.features
        self.feat_channels = backbone.classifier[0].in_features  # 960
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.drop2d  = nn.Dropout2d(dropout2d)
        flat_dim = self.feat_channels * 2 * 2  # 960*4 = 3840
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.GELU(), nn.Dropout(proj_dropout),
            nn.Linear(256, output_dim), nn.GELU()
        )
    def forward(self, x):  # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.drop2d(x)
        x = x.view(B*T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)  # (B, T, 512)

class EngagementModelNoFusion(nn.Module):
    def __init__(self, cnn_feat_dim=512, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size=cnn_feat_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # logit
        )
    def forward(self, cnn_feats):
        _, (hn, _) = self.lstm(cnn_feats)  # hn: (1, B, H)
        x = hn.squeeze(0)                  # (B, H)
        return self.fc(x)                  # (B, 1)

# ===== 체크포인트 로드 =====
def load_checkpoint_and_models(ckpt_path: str):
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    thr_acc = ckpt.get("thr_acc", None)
    thr_rec = ckpt.get("thr_rec", None)
    cnn = CNNEncoder().to(DEVICE)
    model = EngagementModelNoFusion().to(DEVICE)
    cnn.load_state_dict(ckpt["cnn_state_dict"])
    model.load_state_dict(ckpt["model_state_dict"])
    cnn.eval(); model.eval()
    return cnn, model, thr_acc, thr_rec

# ===== 추론 =====
@torch.no_grad()
def infer_clip(cnn, model, frames_tensor_list, threshold: float = 0.5):
    x = torch.stack(frames_tensor_list, dim=0)  # (T,3,H,W)
    x = x.unsqueeze(0).to(DEVICE)               # (1,T,3,H,W)
    feats = cnn(x)                              # (1,T,512)
    logit = model(feats)                        # (1,1)
    p = torch.sigmoid(logit)[0, 0].item()       # focused 확률
    label = "focused" if p >= threshold else "unfocused"
    conf = p if label == "focused" else 1 - p   # 현재 라벨의 확률
    return label, conf, p

# ===== 메인 =====
def main():
    # 모델/임계값
    cnn, model, thr_acc, thr_rec = load_checkpoint_and_models(CKPT_PATH)
    threshold = float(thr_acc) if thr_acc is not None else (float(thr_rec) if thr_rec is not None else 0.5)
    print(f"[INFO] threshold = {threshold:.3f}")

    # 장치/얼굴 검출기
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1,               # 0: 근거리, 1: 일반
        min_detection_confidence=0.5
    )

    buffer = deque(maxlen=T_WINDOW)
    print("실시간 추론 시작 (3 FPS 입력, 5초 간격 업데이트). 종료: q 키")

    # 3 FPS 타이밍 제어
    interval = 1.0 / TARGET_FPS
    next_t = time.time()

    # 슬라이딩 윈도우 추론: 5초(15프레임)마다 갱신
    frame_idx = 0
    last_label, last_conf, last_p = "preparing", 0.0, None
    prev_bbox = None

    try:
        while True:
            # 3 FPS 동기화
            now = time.time()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += interval

            ok, frame_bgr = cap.read()
            if not ok:
                print("프레임 획득 실패")
                break

            # 얼굴 crop (+ bbox 반환). 탐지 실패 시 prev_bbox로 대체/혹은 전체 프레임.
            face_rgb, curr_bbox = crop_face(
                frame_bgr, face_detector, fallback_to_full=True, prev_bbox=prev_bbox, scale=0.5, margin=0.12
            )
            if curr_bbox is not None:
                prev_bbox = curr_bbox

            if face_rgb is None:
                cv2.putText(frame_bgr, "No face", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Engagement (Real-time)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 전처리 → 버퍼 적재
            pil = Image.fromarray(face_rgb)
            tensor_3hw = preprocess(pil)
            buffer.append(tensor_3hw)

            frame_idx += 1

            # 추론 타이밍: 버퍼가 가득(30프레임)이고, 15프레임(≈5초)마다
            if len(buffer) == T_WINDOW and (frame_idx % STRIDE_FRAMES == 0):
                try:
                    label, conf, p = infer_clip(cnn, model, list(buffer), threshold)
                    last_label, last_conf, last_p = label, conf, p
                except Exception as e:
                    print("Inference error:", e)
                    label, conf, p = last_label, last_conf, last_p
            else:
                # 갱신 타이밍이 아니면 직전 결과 유지
                label, conf, p = last_label, last_conf, last_p

            # 색상
            color = (
                (0, 200, 0) if label in ("focused", "집중함")
                else (0, 0, 255) if label in ("unfocused", "안함")
                else (180, 180, 0)
            )

            # 얼굴 bbox만 그리기
            if prev_bbox is not None:
                x1, y1, x2, y2 = prev_bbox
                h, w = frame_bgr.shape[:2]
                x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)

            # 현재 라벨 확률만 표시
            if p is not None:
                conf_to_show = p if label in ("focused", "집중함") else (1 - p if label in ("unfocused", "안함") else None)
            else:
                conf_to_show = None

            if conf_to_show is not None:
                text = f"{label}  {conf_to_show*100:.1f}%  thr={threshold:.2f}"
            else:
                text = f"{label}  thr={threshold:.2f}"

            cv2.putText(frame_bgr, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Engagement (Real-time)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
