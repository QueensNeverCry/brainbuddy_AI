import torch
import cv2
import numpy as np
from collections import deque
import time
import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from models.pytorch_concentration import create_model
from utils.face_detector import FaceDetector
from utils.attention_features import AttentionFeatureExtractor

class PyTorchConcentrationInference:
    """PyTorch 모델 기반 실시간 집중도 분석"""
    
    def __init__(self, model_path, model_type='lstm', device='auto'):
        # 디바이스 설정 (GPU 자동 감지)
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model_type = model_type
        
        # 모델 로드
        self.load_model(model_path)
        
        # 얼굴 검출 및 특징 추출기
        self.face_detector = FaceDetector()
        self.feature_extractor = AttentionFeatureExtractor()
        
        # 30프레임 버퍼
        self.frame_buffer = deque(maxlen=30)
        self.analysis_results = []
        
        # UI 설정
        self.cls_name = {0: 'Not Focused', 1: 'Focused'}
        self.cls_color = {0: (0, 0, 255), 1: (0, 255, 0)}
        
        # 직사각형 집중 영역
        self.attention_zone_width = 720
        self.attention_zone_height = 180
        
        print(f"✅ PyTorch 집중도 분석기 초기화 완료")
        print(f"모델: {model_type}")
        print(f"디바이스: {self.device}")
        
        # GPU 메모리 정보 출력
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU 메모리: {gpu_memory:.1f}GB")
            
    def load_model(self, model_path):
        """PyTorch 모델 로드"""
        print(f"📂 모델 로드: {model_path}")
        
        # CPU에서 로드 후 디바이스로 이동
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 모델 생성
        self.model = create_model(self.model_type, input_dim=31)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 모델 정보
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"📊 모델 파라미터: {total_params:,}개")
        print(f"✅ 모델 로드 완료 (F1: {checkpoint.get('val_f1', 'N/A'):.4f})")
        
        # GPU 사용 시 워밍업
        if self.device.type == 'cuda':
            self.warmup_gpu()
    
    def warmup_gpu(self):
        """GPU 워밍업 (첫 추론 속도 최적화)"""
        print("🔥 GPU 워밍업 중...")
        dummy_input = torch.randn(1, 30, 31).to(self.device)
        
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print("✅ GPU 워밍업 완료")
    
    def extract_frame_features(self, frame):
        """프레임에서 특징 추출"""
        # 얼굴 검출
        face_box = self.face_detector.detect_face(frame)
        
        # 특징 추출 (26차원 + 5차원 attention = 31차원)
        features, attention_features = self.feature_extractor.extract_features(frame, face_box)
        
        # 특징 결합
        combined_features = np.concatenate([
            features,
            [
                attention_features['central_focus'],
                attention_features['gaze_fixation'],
                attention_features['head_stability'],
                attention_features['face_orientation'],
                attention_features['attention_score']
            ]
        ])
        
        return combined_features, face_box, attention_features
    
    def predict_concentration(self):
        """30프레임으로 집중도 예측"""
        if len(self.frame_buffer) < 30:
            return None, 0.0
        
        # 30프레임을 텐서로 변환
        sequence = torch.FloatTensor(list(self.frame_buffer)).unsqueeze(0)  # [1, 30, 31]
        sequence = sequence.to(self.device)
        
        # 성능 측정
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        # 모델 예측
        with torch.no_grad():
            output = self.model(sequence)
            confidence = output.item()
            prediction = 1 if confidence > 0.5 else 0
        
        # 추론 시간 계산
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.time() - start_time
        
        return prediction, confidence, inference_time
    
    def run_realtime_analysis(self):
        """실시간 분석 실행"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다")
            return
        
        frame_count = 0
        analysis_count = 0
        start_time = time.time()
        
        # 성능 모니터링
        inference_times = deque(maxlen=10)
        
        print("🚀 PyTorch 실시간 집중도 분석 시작")
        print("30프레임이 쌓이면 분석을 시작합니다...")
        print("ESC 키로 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 프레임별 특징 추출
            features, face_box, attention_features = self.extract_frame_features(frame)
            self.frame_buffer.append(features)
            
            # 30프레임마다 분석
            if frame_count % 30 == 0 and len(self.frame_buffer) == 30:
                analysis_count += 1
                prediction, confidence, inference_time = self.predict_concentration()
                
                if prediction is not None:
                    inference_times.append(inference_time)
                    
                    # 결과 저장
                    result = {
                        'frame': frame_count,
                        'analysis': analysis_count,
                        'prediction': prediction,
                        'confidence': confidence,
                        'inference_time': inference_time,
                        'timestamp': time.time()
                    }
                    self.analysis_results.append(result)
                    
                    # 결과 출력 (성능 정보 포함)
                    status = "🎯 집중" if prediction == 1 else "❌ 비집중"
                    avg_inference_time = np.mean(list(inference_times)) if inference_times else 0
                    
                    print(f"[분석 {analysis_count:3d}] {status} | "
                          f"확신도: {confidence:.3f} | "
                          f"추론시간: {inference_time*1000:.1f}ms | "
                          f"평균: {avg_inference_time*1000:.1f}ms")
            
            # UI 그리기
            self.draw_ui(frame, face_box, attention_features, inference_times)
            
            # FPS 및 성능 표시
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # GPU 메모리 사용량 표시 (CUDA인 경우)
            if self.device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                cv2.putText(frame, f"GPU Mem: {memory_used:.0f}MB", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 버퍼 상태
            buffer_status = f"Buffer: {len(self.frame_buffer)}/30"
            cv2.putText(frame, buffer_status, (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("PyTorch Concentration Analysis", frame)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        # 결과 요약
        self.print_summary(start_time, inference_times)
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_ui(self, frame, face_box, attention_features, inference_times):
        """UI 그리기 (성능 정보 포함)"""
        # 최근 분석 결과 표시
        if self.analysis_results:
            recent_result = self.analysis_results[-1]
            prediction = recent_result['prediction']
            confidence = recent_result['confidence']
            
            color = self.cls_color[prediction]
            status_text = "FOCUSED" if prediction == 1 else "NOT FOCUSED"
            
            cv2.putText(frame, f"Status: {status_text}", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.3f}", (10, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 얼굴 박스
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 집중 영역 (직사각형)
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        rect_left = center_x - self.attention_zone_width // 2
        rect_right = center_x + self.attention_zone_width // 2
        rect_top = center_y - self.attention_zone_height // 2
        rect_bottom = center_y + self.attention_zone_height // 2
        
        cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (255, 255, 255), 2)
        
        # 성능 정보
        avg_inference_time = np.mean(list(inference_times)) if inference_times else 0
        cv2.putText(frame, f"Avg Inference: {avg_inference_time*1000:.1f}ms", (10, frame.shape[0] - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 모델 정보
        cv2.putText(frame, f"Model: PyTorch {self.model_type.upper()}", (10, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Device: {self.device}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def print_summary(self, start_time, inference_times):
        """결과 요약 (성능 포함)"""
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"📊 PyTorch 실시간 분석 결과")
        print(f"{'='*70}")
        print(f"모델: {self.model_type}")
        print(f"디바이스: {self.device}")
        print(f"총 실행 시간: {total_time:.1f}초")
        print(f"총 분석 횟수: {len(self.analysis_results)}번")
        
        if inference_times:
            avg_inference_time = np.mean(list(inference_times))
            min_inference_time = np.min(list(inference_times))
            max_inference_time = np.max(list(inference_times))
            
            print(f"평균 추론 시간: {avg_inference_time*1000:.2f}ms")
            print(f"최소 추론 시간: {min_inference_time*1000:.2f}ms")
            print(f"최대 추론 시간: {max_inference_time*1000:.2f}ms")
        
        if self.analysis_results:
            focus_count = sum(1 for r in self.analysis_results if r['prediction'] == 1)
            focus_ratio = focus_count / len(self.analysis_results) * 100
            
            print(f"집중 판정: {focus_count}/{len(self.analysis_results)}번 ({focus_ratio:.1f}%)")
            
            avg_confidence = np.mean([r['confidence'] for r in self.analysis_results])
            print(f"평균 확신도: {avg_confidence:.3f}")
        
        print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch 집중도 모델 실시간 추론')
    parser.add_argument('--model', type=str, required=True,
                       help='PyTorch 모델 파일 경로 (.pt)')
    parser.add_argument('--model_type', type=str, default='lstm',
                       choices=['lstm', 'transformer', 'cnn1d'],
                       help='모델 타입')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='디바이스 선택 (auto: 자동감지)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        return
    
    # 추론기 생성 및 실행
    inference = PyTorchConcentrationInference(
        model_path=args.model, 
        model_type=args.model_type,
        device=args.device
    )
    
    try:
        inference.run_realtime_analysis()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")

if __name__ == "__main__":
    main()
