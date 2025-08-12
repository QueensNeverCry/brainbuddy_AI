# real_time_webcam_test.py (실시간 웹캠 집중도 모니터링)
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
import threading
import queue

# ------------------ 모델 클래스들 (기존과 동일) ------------------
class CNNEncoderV1(nn.Module):
    def __init__(self, output_dim=1280):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(
            nn.Linear(1280 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, output_dim),
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

class EngagementModelV1(nn.Module):
    def __init__(self, cnn_feat_dim=1280, fusion_feat_dim=5, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        self.input_projection = nn.Linear(cnn_feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(d_model + fusion_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, cnn_feats, fusion_feats):
        x = self.input_projection(cnn_feats)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        transformer_out = self.transformer_encoder(x)
        pooled = self.global_pool(transformer_out.transpose(1, 2)).squeeze(-1)
        combined = torch.cat([pooled, fusion_feats], dim=1)
        return self.fc(combined)

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

class TransformerEnsembleModel(nn.Module):
    def __init__(self, cnn_v1, model_v1, cnn_v2, model_v2, ensemble_method='learned'):
        super().__init__()
        self.cnn_v1 = cnn_v1
        self.model_v1 = model_v1
        self.cnn_v2 = cnn_v2
        self.model_v2 = model_v2
        self.ensemble_method = ensemble_method
        
        if ensemble_method == 'weighted':
            self.register_buffer('weights', torch.tensor([0.3, 0.7]))
        elif ensemble_method == 'learned':
            self.ensemble_weights = nn.Parameter(torch.tensor([0.3, 0.7]))
            self.ensemble_fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )

    def forward(self, videos, fusion_feats):
        feats_v1 = self.cnn_v1(videos)
        feats_v2 = self.cnn_v2(videos)
        
        logits_v1 = self.model_v1(feats_v1, fusion_feats)
        logits_v2 = self.model_v2(feats_v2, fusion_feats)
        
        if self.ensemble_method == 'weighted':
            return self.weights[0] * logits_v1 + self.weights[1] * logits_v2
        elif self.ensemble_method == 'learned':
            prob_v1 = torch.sigmoid(logits_v1)
            prob_v2 = torch.sigmoid(logits_v2)
            normalized_weights = torch.softmax(self.ensemble_weights, dim=0)
            weighted_v1 = prob_v1 * normalized_weights[0]
            weighted_v2 = prob_v2 * normalized_weights[1]
            combined_input = torch.cat([weighted_v1, weighted_v2], dim=1)
            return self.ensemble_fc(combined_input)

# ------------------ 실시간 웹캠 클래스 ------------------
class RealTimeEngagementMonitor:
    def __init__(self, model_path, device='cuda', threshold=0.7):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.frame_buffer = deque(maxlen=30)  # 30프레임 버퍼
        self.result_queue = queue.Queue(maxsize=10)
        
        # 변환 설정
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 모델 로드
        self.load_model(model_path)
        
        # 결과 통계
        self.recent_predictions = deque(maxlen=10)  # 최근 10개 예측 평균
        self.total_frames = 0
        self.focused_frames = 0
        
        print(f"🚀 실시간 집중도 모니터 초기화 완료!")
        print(f"   - 디바이스: {self.device}")
        print(f"   - 임계값: {self.threshold}")
        print(f"   - 모델 로드: 성공")
    
    def load_model(self, model_path):
        """앙상블 모델 로드"""
        print(f"📂 모델 로딩 중: {model_path}")
        
        # 개별 모델들 초기화
        cnn_v1 = CNNEncoderV1().to(self.device)
        model_v1 = EngagementModelV1(d_model=128, nhead=8, num_layers=3).to(self.device)
        cnn_v2 = CNNEncoderV2().to(self.device)
        model_v2 = EngagementModelV2(d_model=256, nhead=8, num_layers=4).to(self.device)
        
        # 개별 모델 가중치 로드
        v1_checkpoint = torch.load("./log/best_model2.pt", map_location=self.device)
        cnn_v1.load_state_dict(v1_checkpoint['cnn_state_dict'])
        model_v1.load_state_dict(v1_checkpoint['model_state_dict'])
        
        v2_checkpoint = torch.load("./log/v2/best_model_v2.pt", map_location=self.device)
        cnn_v2.load_state_dict(v2_checkpoint['cnn_state_dict'])
        model_v2.load_state_dict(v2_checkpoint['model_state_dict'])
        
        # 앙상블 모델 생성 및 로드
        self.model = TransformerEnsembleModel(
            cnn_v1, model_v1, cnn_v2, model_v2, 
            ensemble_method='learned'
        ).to(self.device)
        
        # 앙상블 가중치 로드
        ensemble_checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ensemble_checkpoint['ensemble_state_dict'])
        
        # 모델을 evaluation 모드로 설정
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 모델 정보 출력
        if 'accuracy' in ensemble_checkpoint:
            accuracy = ensemble_checkpoint['accuracy']
            print(f"✅ 앙상블 모델 로드 완료 (정확도: {accuracy:.1%})")
    
    def preprocess_frame(self, frame):
        """프레임 전처리"""
        # BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # PIL Image로 변환
        pil_image = Image.fromarray(frame_rgb)
        # 변환 적용
        tensor = self.transform(pil_image)
        return tensor
    
    def predict_engagement(self):
        """현재 프레임 버퍼로 집중도 예측"""
        if len(self.frame_buffer) < 30:
            return None, None
        
        # 30프레임을 텐서로 변환
        frames = torch.stack(list(self.frame_buffer))  # (30, 3, 224, 224)
        frames = frames.unsqueeze(0).to(self.device)   # (1, 30, 3, 224, 224)
        
        # 더미 fusion features (실시간에서는 사용하지 않음)
        fusion = torch.zeros(1, 5).to(self.device)
        
        # 예측 수행
        with torch.no_grad():
            logits = self.model(frames, fusion)
            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability > self.threshold else 0
        
        return prediction, probability
    
    def update_statistics(self, prediction):
        """통계 업데이트"""
        if prediction is not None:
            self.recent_predictions.append(prediction)
            self.total_frames += 1
            if prediction == 0:  # 집중함
                self.focused_frames += 1
    
    def get_current_stats(self):
        """현재 통계 반환"""
        if len(self.recent_predictions) == 0:
            return 0, 0, 0
        
        recent_focus_rate = (len(self.recent_predictions) - sum(self.recent_predictions)) / len(self.recent_predictions)
        overall_focus_rate = self.focused_frames / max(self.total_frames, 1)
        current_prediction = self.recent_predictions[-1] if self.recent_predictions else 0
        
        return recent_focus_rate, overall_focus_rate, current_prediction
    
    def draw_overlay(self, frame, prediction, probability):
        """결과를 프레임에 오버레이"""
        height, width = frame.shape[:2]
        
        # 현재 통계 가져오기
        recent_focus, overall_focus, current_pred = self.get_current_stats()
        
        # 배경 사각형 그리기
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 집중 상태에 따른 색상 설정
        if prediction == 0:  # 집중함
            color = (0, 255, 0)  # 초록색
            status = "FOCUSED"
            status_ko = "집중함"
        else:  # 집중하지 않음
            color = (0, 0, 255)  # 빨간색
            status = "UNFOCUSED"
            status_ko = "집중하지않음"
        
        # 텍스트 정보 표시
        cv2.putText(frame, f"Status: {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Korean: {status_ko}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Probability: {probability:.3f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Recent Focus: {recent_focus:.1%}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Overall Focus: {overall_focus:.1%}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 우측에 집중도 바 그리기
        bar_x = width - 60
        bar_y = 50
        bar_height = 200
        bar_width = 30
        
        # 배경 바
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        
        # 집중도 바 (최근 집중도 기준)
        fill_height = int(bar_height * recent_focus)
        cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height), 
                     (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
        
        # 바 테두리
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        cv2.putText(frame, "Focus", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_id=0, save_video=False):
        """실시간 웹캠 모니터링 실행"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ 카메라 {camera_id}를 열 수 없습니다.")
            return
        
        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 비디오 저장 설정 (선택사항)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('engagement_monitor.avi', fourcc, 20.0, (640, 480))
        
        print("🎥 실시간 집중도 모니터링 시작!")
        print("   - 'q' 키를 눌러 종료")
        print("   - 'r' 키를 눌러 통계 리셋")
        print("   - 'p' 키를 눌러 예측 일시정지/재개")
        
        frame_count = 0
        prediction_active = True
        last_prediction_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 프레임을 버퍼에 추가
                processed_frame = self.preprocess_frame(frame)
                self.frame_buffer.append(processed_frame)
                
                # 3프레임마다 예측 수행 (속도 최적화)
                prediction, probability = None, None
                if prediction_active and frame_count % 3 == 0 and len(self.frame_buffer) == 30:
                    current_time = time.time()
                    prediction, probability = self.predict_engagement()
                    prediction_time = time.time() - current_time
                    
                    if prediction is not None:
                        self.update_statistics(prediction)
                        last_prediction_time = current_time
                        print(f"Frame {frame_count}: {'집중함' if prediction == 0 else '집중하지않음'} "
                              f"(확률: {probability:.3f}, 처리시간: {prediction_time:.3f}초)")
                
                # 마지막 예측 결과로 오버레이 (예측이 없으면 이전 결과 사용)
                if hasattr(self, '_last_prediction'):
                    display_pred, display_prob = self._last_prediction
                else:
                    display_pred, display_prob = prediction, probability
                
                if prediction is not None:
                    self._last_prediction = (prediction, probability)
                    display_pred, display_prob = prediction, probability
                
                # 오버레이 그리기
                if display_pred is not None:
                    frame = self.draw_overlay(frame, display_pred, display_prob)
                
                # 예측 상태 표시
                status_text = "ACTIVE" if prediction_active else "PAUSED"
                cv2.putText(frame, f"Prediction: {status_text}", (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 화면에 표시
                cv2.imshow('Real-time Engagement Monitor', frame)
                
                # 비디오 저장
                if save_video:
                    out.write(frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 통계 리셋
                    self.recent_predictions.clear()
                    self.total_frames = 0
                    self.focused_frames = 0
                    print("📊 통계가 리셋되었습니다.")
                elif key == ord('p'):
                    # 예측 일시정지/재개
                    prediction_active = not prediction_active
                    print(f"🔄 예측 {'재개' if prediction_active else '일시정지'}")
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중단되었습니다.")
        
        finally:
            # 최종 통계 출력
            recent_focus, overall_focus, _ = self.get_current_stats()
            print(f"\n📊 최종 통계:")
            print(f"   - 총 처리 프레임: {self.total_frames}")
            print(f"   - 전체 집중률: {overall_focus:.1%}")
            print(f"   - 최근 집중률: {recent_focus:.1%}")
            
            # 정리
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()

# ------------------ 메인 실행 함수 ------------------
def main():
    print("🚀 실시간 웹캠 집중도 테스트 시작")
    print("="*50)
    
    # 모델 경로 설정 (최신 앙상블 모델 사용)
    model_paths = [
        "./log/ensemble/best_speed_ensemble.pt",          # 속도 최적화 앙상블
        "./log/ensemble/best_weighted_ensemble.pt",       # 가중 앙상블
        "./log/ensemble/best_transformer_ensemble.pt",    # Transformer 앙상블
    ]
    
    # 사용할 모델 찾기
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ 앙상블 모델을 찾을 수 없습니다. 다음 경로들을 확인하세요:")
        for path in model_paths:
            print(f"   - {path}")
        return
    
    print(f"📂 사용할 모델: {model_path}")
    
    # 설정 옵션
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = 0.7  # 임계값 (Version 2 테스트에서 최적값)
    camera_id = 0    # 웹캠 ID (보통 0이 기본 웹캠)
    save_video = False  # 비디오 저장 여부
    
    try:
        # 실시간 모니터 초기화
        monitor = RealTimeEngagementMonitor(
            model_path=model_path,
            device=device,
            threshold=threshold
        )
        
        # 실시간 모니터링 시작
        monitor.run(camera_id=camera_id, save_video=save_video)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
