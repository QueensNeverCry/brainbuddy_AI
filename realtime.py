# simple_face_crop_webcam.py (간략한 UI + 개선된 얼굴 크롭)
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import time
import os
import math
from collections import deque

# MediaPipe 설치 확인
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe 사용 가능")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe 미설치")

# ------------------ 모델 클래스들 ------------------
class CNNEncoderV2(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(B * T, -1)
        x = self.fc(x)
        return x.view(B, T, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :].to(x.device)

class EngagementModelV2(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(cnn_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.15,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2 + fusion_feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        x = self.input_projection(cnn_feats)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        
        avg_pooled = self.global_avg_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        max_pooled = self.global_max_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        combined = torch.cat([pooled, fusion_feats], dim=1)
        return self.fc(combined)

# ------------------ 개선된 얼굴 크롭 클래스 ------------------
class ImprovedFaceCropper:
    """개선된 MediaPipe 얼굴 크롭 (거리 문제 해결)"""
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            # model_selection=1로 설정 (더 넓은 범위, 5m까지)
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # ✅ 0 → 1로 변경 (더 넓은 탐지 범위)
                min_detection_confidence=0.3  # ✅ 0.5 → 0.3으로 낮춤 (더 민감하게)
            )
            print("✅ 개선된 MediaPipe FaceDetection 초기화")
        else:
            self.face_detection = None
            print("⚠️ MediaPipe 없이 중앙 크롭 사용")
    
    def crop_face(self, frame, padding_ratio=0.5):  # ✅ 패딩 0.3 → 0.5로 증가
        """개선된 얼굴 크롭 (거리 문제 해결)"""
        if not MEDIAPIPE_AVAILABLE or self.face_detection is None:
            return self._adaptive_center_crop(frame), False
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 탐지
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            # 가장 신뢰도 높은 얼굴 선택
            detection = max(results.detections, 
                          key=lambda x: x.score)  # ✅ size 대신 confidence 기준
            
            bbox = detection.location_data.relative_bounding_box
            
            # 절대 좌표 변환
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            face_w = int(bbox.width * w)
            face_h = int(bbox.height * h)
            
            # ✅ 더 큰 패딩 적용 (어깨까지 포함)
            padding_w = int(face_w * padding_ratio)
            padding_h = int(face_h * padding_ratio)
            
            # 크롭 영역 계산
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(w, x + face_w + padding_w)
            y2 = min(h, y + face_h + padding_h)
            
            # ✅ 정사각형 만들기 (더 관대하게)
            crop_w = x2 - x1
            crop_h = y2 - y1
            target_size = max(crop_w, crop_h)  # 더 큰 쪽에 맞춤
            
            # 중앙 정렬
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            half_size = target_size // 2
            x1 = max(0, center_x - half_size)
            y1 = max(0, center_y - half_size)
            x2 = min(w, center_x + half_size)
            y2 = min(h, center_y + half_size)
            
            # 경계 조정
            if x2 - x1 < target_size:
                if x1 == 0:
                    x2 = min(w, x1 + target_size)
                else:
                    x1 = max(0, x2 - target_size)
            
            if y2 - y1 < target_size:
                if y1 == 0:
                    y2 = min(h, y1 + target_size)
                else:
                    y1 = max(0, y2 - target_size)
            
            cropped_face = frame[y1:y2, x1:x2]
            
            if cropped_face.shape[0] > 0 and cropped_face.shape[1] > 0:
                cropped_face = cv2.resize(cropped_face, (224, 224))
                return cropped_face, True
        
        # 얼굴 탐지 실패 시 적응형 중앙 크롭
        return self._adaptive_center_crop(frame), False
    
    def _adaptive_center_crop(self, frame):
        """적응형 중앙 크롭 (상체 포함)"""
        h, w, _ = frame.shape
        
        # ✅ 더 큰 크롭 비율 (상체 포함)
        crop_ratio = 0.8  # 화면의 80% 사용
        crop_size = int(min(h, w) * crop_ratio)
        
        # 중앙에서 약간 위쪽으로 이동 (머리가 중앙에 오도록)
        center_x = w // 2
        center_y = int(h * 0.4)  # ✅ 중앙보다 위쪽
        
        half_size = crop_size // 2
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)
        
        cropped = frame[y1:y2, x1:x2]
        return cv2.resize(cropped, (224, 224))

# ------------------ 간략한 UI 모니터 ------------------
class SimpleFocusMonitor:
    def __init__(self, model_path, device='cuda', threshold=0.7):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.frame_buffer = deque(maxlen=30)
        
        # 얼굴 크롭 초기화
        self.face_cropper = ImprovedFaceCropper()
        
        # 전처리
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.load_model(model_path)
        
        # 통계 (간단하게)
        self.recent_predictions = deque(maxlen=5)  # 5개만 사용
        
        print(f"🚀 간략한 집중도 모니터 초기화 완료!")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 임계값: {self.threshold}")
    
    def load_model(self, model_path):
        """Version 2 모델 로드"""
        print(f"📂 모델 로딩 중: {model_path}")
        
        self.cnn = CNNEncoderV2().to(self.device)
        self.model = EngagementModelV2(d_model=256, nhead=8, num_layers=4).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.cnn.eval()
        self.model.eval()
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False
            
        print("✅ 모델 로드 완료")
    
    def preprocess_frame(self, frame):
        """얼굴 크롭 + 전처리"""
        # 개선된 얼굴 크롭
        cropped_face, face_detected = self.face_cropper.crop_face(frame)
        
        # BGR to RGB
        face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        
        # 텐서 변환
        tensor = self.transform(face_rgb)
        
        return tensor, face_detected
    
    def predict_engagement(self):
        """집중도 예측"""
        if len(self.frame_buffer) < 30:
            return None, None
        
        # 30프레임 스택
        frames_data = list(self.frame_buffer)
        frames = torch.stack([data[0] for data in frames_data])
        frames = frames.unsqueeze(0).to(self.device)
        
        # 더미 fusion features
        fusion = torch.zeros(1, 5).to(self.device)
        
        # 예측
        with torch.no_grad():
            cnn_features = self.cnn(frames)
            logits = self.model(cnn_features, fusion)
            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability > self.threshold else 0
        
        return prediction, probability
    
    def draw_simple_overlay(self, frame, prediction, probability):
        """간략한 UI 오버레이"""
        height, width = frame.shape[:2]
        
        # 최근 집중률 계산
        focus_rate = 0
        if len(self.recent_predictions) > 0:
            focus_rate = (len(self.recent_predictions) - sum(self.recent_predictions)) / len(self.recent_predictions)
        
        # ✅ 간단한 배경 (작게)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # ✅ 집중 상태 (큰 글씨)
        if prediction == 0:
            color = (0, 255, 0)
            status = "집중함"
        else:
            color = (0, 0, 255)
            status = "집중안함"
        
        # ✅ 메인 정보만 표시
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"{probability*100:.0f}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # ✅ 작은 집중도 바 (우측)
        bar_x = width - 40
        bar_y = 20
        bar_height = 120
        bar_width = 20
        
        # 바 배경
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # 집중도 표시
        if prediction == 0:  # 집중함
            fill_color = (0, 255, 0)
            fill_height = int(bar_height * (1 - probability))  # 확률이 낮을수록 집중함
        else:  # 집중안함
            fill_color = (0, 0, 255)
            fill_height = int(bar_height * probability)  # 확률이 높을수록 집중안함
        
        cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height), 
                     (bar_x + bar_width, bar_y + bar_height), fill_color, -1)
        
        # 바 테두리
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        return frame
    
    def run(self, camera_id=0):
        """간략한 실시간 모니터링"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("🎥 간략한 집중도 모니터링 시작!")
        print("   - 'q' 키로 종료")
        print("   - 'r' 키로 리셋")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 얼굴 크롭 및 전처리
                processed_frame, face_detected = self.preprocess_frame(frame)
                self.frame_buffer.append((processed_frame, face_detected))
                
                # 3프레임마다 예측
                prediction, probability = None, None
                if frame_count % 3 == 0 and len(self.frame_buffer) == 30:
                    prediction, probability = self.predict_engagement()
                    
                    if prediction is not None:
                        self.recent_predictions.append(prediction)
                        status = "집중함" if prediction == 0 else "집중안함"
                        print(f"Frame {frame_count}: {status} ({probability*100:.0f}%)")
                
                # 이전 결과 유지
                if hasattr(self, '_last_result'):
                    display_pred, display_prob = self._last_result
                else:
                    display_pred, display_prob = prediction, probability
                
                if prediction is not None:
                    self._last_result = (prediction, probability)
                    display_pred, display_prob = prediction, probability
                
                # ✅ 간략한 오버레이 그리기
                if display_pred is not None:
                    frame = self.draw_simple_overlay(frame, display_pred, display_prob)
                
                cv2.imshow('Focus Monitor', frame)
                
                # 키 입력
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.recent_predictions.clear()
                    print("📊 리셋 완료")
                
        except KeyboardInterrupt:
            print("\n⏹️ 중단됨")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

# ------------------ 메인 실행 ------------------
def main():
    print("🚀 간략한 얼굴 크롭 집중도 테스트")
    print("="*40)
    
    model_paths = [
        "./log/v2/best_model_v2.pt",
        "./log/best_model2.pt",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 모델을 찾을 수 없습니다.")
        return
    
    print(f"📂 모델: {model_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = 0.7
    
    try:
        monitor = SimpleFocusMonitor(
            model_path=model_path,
            device=device,
            threshold=threshold
        )
        
        monitor.run(camera_id=0)
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
