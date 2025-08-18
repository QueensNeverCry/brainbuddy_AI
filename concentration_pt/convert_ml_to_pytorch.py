import torch
import torch.nn as nn
import numpy as np
import pickle
import cv2
import os
from collections import deque
import math
from tqdm import tqdm
import sys
sys.path.append('.')

from utils.face_detector import FaceDetector
from utils.attention_features import AttentionFeatureExtractor

class MLToPyTorchConverter:
    """ML 모델을 PyTorch로 변환하고 데이터 생성"""
    
    def __init__(self, ml_model_path):
        """
        Args:
            ml_model_path: 기존 .pkl 모델 경로
        """
        self.ml_model_path = ml_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 기존 ML 모델 로드
        self.load_ml_model()
        
        # 얼굴 검출 및 특징 추출기
        self.face_detector = FaceDetector()
        self.feature_extractor = AttentionFeatureExtractor()
        
        print(f"✅ ML→PyTorch 변환기 초기화 완료")
        print(f"디바이스: {self.device}")
    
    def load_ml_model(self):
        """기존 ML 모델 로드 (다양한 방법 시도)"""
        try:
            # 방법 1: joblib로 로드 (XGBoost 모델에 일반적)
            import joblib
            loaded_data = joblib.load(self.ml_model_path)
            
            # 로드된 데이터가 딕셔너리인지 확인
            if isinstance(loaded_data, dict):
                print("⚠️ 모델이 딕셔너리로 저장되었습니다. 모델 객체를 추출합니다.")
                
                # 딕셔너리에서 모델 객체 찾기
                if 'model' in loaded_data:
                    self.ml_model = loaded_data['model']
                elif 'estimator' in loaded_data:
                    self.ml_model = loaded_data['estimator']
                elif 'clf' in loaded_data:
                    self.ml_model = loaded_data['clf']
                elif 'xgb_model' in loaded_data:
                    self.ml_model = loaded_data['xgb_model']
                else:
                    # 딕셔너리의 첫 번째 값이 모델인지 확인
                    for key, value in loaded_data.items():
                        if hasattr(value, 'predict') and hasattr(value, 'predict_proba'):
                            self.ml_model = value
                            print(f"📂 딕셔너리에서 모델 객체 발견: {key}")
                            break
                    else:
                        raise ValueError("딕셔너리에서 유효한 모델 객체를 찾을 수 없습니다")
            else:
                # 직접 모델 객체인 경우
                self.ml_model = loaded_data
            
            print(f"📂 ML 모델 로드 성공: {type(self.ml_model)}")
            
            # 모델이 predict와 predict_proba 메서드를 가지고 있는지 확인
            if not (hasattr(self.ml_model, 'predict') and hasattr(self.ml_model, 'predict_proba')):
                raise ValueError(f"로드된 객체에 predict 메서드가 없습니다: {type(self.ml_model)}")
            
            print(f"✅ 모델 검증 완료")
            
        except Exception as e:
            print(f"❌ 모든 로드 방법 실패: {e}")
            print("🔄 시뮬레이션 데이터로 대체합니다")
            self.ml_model = None  # 시뮬레이션 모드로 전환

    
    def simulate_30_frame_sequence(self, num_sequences=5000):
        """
        30프레임 시퀀스 데이터 시뮬레이션 생성
        
        Args:
            num_sequences: 생성할 시퀀스 수
            
        Returns:
            sequences: 30프레임 시퀀스 리스트
            labels: 각 시퀀스의 라벨 (0: 비집중, 1: 집중)
        """
        print(f"🔄 {num_sequences}개 30프레임 시퀀스 데이터 생성 중...")
        
        sequences = []
        labels = []
        
        # 가상 카메라 설정
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️ 웹캠이 없어서 시뮬레이션 데이터로 대체")
            return self._generate_simulated_data(num_sequences)
        
        frame_count = 0
        sequence_features = []
        
        with tqdm(total=num_sequences, desc="시퀀스 생성") as pbar:
            while len(sequences) < num_sequences:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ 웹캠 읽기 실패, 시뮬레이션 데이터로 대체")
                    cap.release()
                    return self._generate_simulated_data(num_sequences)
                
                frame_count += 1
                
                try:
                    # 얼굴 검출 및 특징 추출
                    face_box = self.face_detector.detect_face(frame)
                    features, attention_features = self.feature_extractor.extract_features(frame, face_box)
                    
                    # ML 모델로 예측 (3클래스) - 수정된 부분
                    if self.ml_model is not None:
                        try:
                            # XGBoost는 predict()와 predict_proba()를 따로 호출
                            ml_pred = self.ml_model.predict(features.reshape(1, -1))
                            ml_probs = self.ml_model.predict_proba(features.reshape(1, -1))
                            
                            # predict는 배열을 반환하므로 첫 번째 요소 추출
                            ml_pred = ml_pred[0] if isinstance(ml_pred, (list, np.ndarray)) else ml_pred
                            ml_probs = ml_probs[0] if len(ml_probs.shape) > 1 else ml_probs
                            
                        except Exception as e:
                            print(f"⚠️ ML 모델 예측 실패: {e}")
                            # 기본값으로 대체
                            ml_pred = 1  
                            ml_probs = np.array([0.4, 0.3, 0.3])
                    else:
                        # ML 모델이 없는 경우 기본값
                        ml_pred = 1
                        ml_probs = np.array([0.4, 0.3, 0.3])
                    
                    # 논문 기반 보정 적용 (기존 inference 로직)
                    binary_result, confidence = self._apply_inference_correction(
                        ml_pred, ml_probs, attention_features
                    )
                    
                    # 특징 + 라벨 저장
                    frame_data = {
                        'features': features,
                        'attention_features': attention_features,
                        'face_detected': face_box is not None,
                        'label': binary_result
                    }
                    sequence_features.append(frame_data)
                    
                    # 30프레임이 모이면 시퀀스 완성
                    if len(sequence_features) == 30:
                        # 시퀀스의 최종 라벨은 마지막 10프레임의 다수결
                        recent_labels = [f['label'] for f in sequence_features[-10:]]
                        final_label = 1 if sum(recent_labels) >= 5 else 0
                        
                        sequences.append(sequence_features.copy())
                        labels.append(final_label)
                        
                        # 버퍼 초기화 (다음 시퀀스 준비)
                        sequence_features = []
                        pbar.update(1)
                        
                except Exception as e:
                    print(f"⚠️ 프레임 처리 오류: {e}")
                    # 오류 발생 시 해당 프레임 건너뛰기
                    continue
                    
                # 너무 오래 걸리면 시뮬레이션으로 전환
                if frame_count > num_sequences * 50:  # 너무 많은 프레임 처리 시
                    print("⏰ 웹캠 처리 시간 초과, 시뮬레이션 데이터로 대체")
                    cap.release()
                    return self._generate_simulated_data(num_sequences)
        
        cap.release()
        
        print(f"✅ 총 {len(sequences)}개 시퀀스 생성 완료")
        focus_ratio = sum(labels) / len(labels) * 100 if labels else 0
        print(f"📊 집중 비율: {focus_ratio:.1f}%")
        
        return sequences, labels

    
    def _apply_inference_correction(self, ml_pred, ml_probs, attention_features):
        """기존 inference.py의 보정 로직 적용"""
        # 기존 focus_sensitive_prediction 로직 재현
        focus_indicators = 0
        
        # ML 모델 예측 가중치
        if ml_pred == 2:  # 집중
            focus_indicators += 4
        elif ml_pred == 0:  # 비집중
            focus_indicators -= 4
        elif ml_probs[2] > 0.25:
            focus_indicators += 2
        
        # 모델 확신도 보정
        if ml_probs[0] > 0.8:  # 비집중 확률 80% 이상
            focus_indicators -= 3
        elif ml_probs[2] > 0.8:  # 집중 확률 80% 이상
            focus_indicators += 3
        
        # 논문 기반 지표들
        if attention_features['central_focus'] > 0.7:
            focus_indicators += 2
        if attention_features['gaze_fixation'] > 0.7:
            focus_indicators += 2
        if attention_features['attention_score'] > 0.5:
            focus_indicators += 2
        if attention_features['head_stability'] > 0.4:
            focus_indicators += 1
        if attention_features['face_orientation'] > 0.3:
            focus_indicators += 1
        
        # 최종 판정
        if focus_indicators >= 3:
            binary_result = 1
            confidence = min(0.95, 0.6 + attention_features['attention_score'] * 0.2 + ml_probs[2] * 0.15)
        else:
            binary_result = 0
            confidence = max(0.5, (ml_probs[0] + ml_probs[1]) / 2)
        
        return binary_result, confidence
    
    def _generate_simulated_data(self, num_sequences):
        """웹캠이 없을 때 시뮬레이션 데이터 생성"""
        print("🔧 시뮬레이션 데이터 생성 중...")
        
        sequences = []
        labels = []
        
        np.random.seed(42)  # 재현 가능한 데이터
        
        for i in tqdm(range(num_sequences), desc="시뮬레이션 시퀀스 생성"):
            sequence_features = []
            
            # 시퀀스 타입 결정 (집중 vs 비집중)
            is_focused_sequence = np.random.choice([0, 1], p=[0.4, 0.6])  # 60% 집중 데이터
            
            for frame_idx in range(30):
                # 기본 특징 벡터 (26차원)
                if is_focused_sequence:
                    # 집중 시퀀스: 안정적인 특징
                    features = self._generate_focused_features()
                    attention_features = {
                        'central_focus': np.random.uniform(0.6, 1.0),
                        'gaze_fixation': np.random.uniform(0.7, 1.0),
                        'head_stability': np.random.uniform(0.6, 0.9),
                        'face_orientation': np.random.uniform(0.5, 1.0),
                        'attention_score': np.random.uniform(0.6, 0.9)
                    }
                else:
                    # 비집중 시퀀스: 불안정한 특징
                    features = self._generate_unfocused_features()
                    attention_features = {
                        'central_focus': np.random.uniform(0.0, 0.4),
                        'gaze_fixation': np.random.uniform(0.0, 0.5),
                        'head_stability': np.random.uniform(0.2, 0.6),
                        'face_orientation': np.random.uniform(0.0, 0.5),
                        'attention_score': np.random.uniform(0.1, 0.5)
                    }
                
                frame_data = {
                    'features': features,
                    'attention_features': attention_features,
                    'face_detected': True,
                    'label': is_focused_sequence
                }
                sequence_features.append(frame_data)
            
            sequences.append(sequence_features)
            labels.append(is_focused_sequence)
        
        return sequences, labels
    
    def _generate_focused_features(self):
        """집중 상태 특징 생성"""
        features = np.zeros(26, dtype=np.float32)
        
        # 안정적인 머리 포즈
        features[0:3] = np.random.normal([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        
        # 중앙 시선
        features[4] = np.random.normal(640, 50)  # 화면 중앙 X
        features[5] = np.random.normal(360, 30)  # 화면 중앙 Y
        
        # 낮은 변동성
        features[13:15] = np.random.uniform([0.3, 0.3], [1.0, 1.0])
        
        # 높은 안정성
        features[15] = np.random.uniform(0.8, 0.95)
        features[16] = np.random.uniform(0.8, 0.9)
        
        # 낮은 떨림
        features[17] = np.random.uniform(3.0, 10.0)
        features[18] = np.random.uniform(0.01, 0.05)
        
        # 긴 고정 응시
        features[19] = np.random.uniform(10, 25)
        
        # 나머지 특징들
        features[3] = np.random.uniform(60, 80)  # 거리
        features[6:10] = [620, 350, 660, 350]  # 눈 위치
        features[10:12] = [0.3, 0.3]  # EAR
        features[12] = np.random.uniform(0, 2)  # 머리 기울기
        features[20] = np.random.uniform(5, 10)  # 고정 시간
        features[21] = np.random.uniform(0.6, 1.0)  # 중앙 집중
        features[22] = np.random.uniform(0.6, 0.8)  # 깜빡임
        
        return features
    
    def _generate_unfocused_features(self):
        """비집중 상태 특징 생성"""
        features = np.zeros(26, dtype=np.float32)
        
        # 불안정한 머리 포즈
        features[0:3] = np.random.normal([2.0, 2.0, 1.5], [1.0, 1.0, 1.0])
        
        # 분산된 시선
        features[4] = np.random.normal(640, 150)  # 더 넓은 분포
        features[5] = np.random.normal(360, 100)
        
        # 높은 변동성
        features[13:15] = np.random.uniform([2.0, 2.0], [5.0, 5.0])
        
        # 낮은 안정성
        features[15] = np.random.uniform(0.2, 0.5)
        features[16] = np.random.uniform(0.3, 0.6)
        
        # 높은 떨림
        features[17] = np.random.uniform(30.0, 60.0)
        features[18] = np.random.uniform(0.2, 0.5)
        
        # 짧은 고정 응시
        features[19] = np.random.uniform(1, 5)
        
        # 나머지 특징들
        features[3] = np.random.uniform(40, 100)
        features[6:10] = np.random.uniform([500, 300, 700, 400], [800, 500, 900, 600])
        features[10:12] = [0.3, 0.3]
        features[12] = np.random.uniform(0, 8)
        features[20] = np.random.uniform(1, 3)
        features[21] = np.random.uniform(0.0, 0.4)
        features[22] = np.random.uniform(0.3, 0.8)
        
        return features
    
    def save_pytorch_dataset(self, sequences, labels, save_path):
        """PyTorch 학습용 데이터셋 저장"""
        print(f"💾 PyTorch 데이터셋 저장: {save_path}")
        
        # 특징들을 텐서로 변환
        processed_sequences = []
        processed_labels = []
        
        for seq, label in tqdm(zip(sequences, labels), desc="데이터 변환", total=len(sequences)):
            # 30프레임의 특징을 스택
            frame_features = []
            for frame_data in seq:
                # 26차원 특징 + 5차원 attention 특징 = 31차원
                combined_features = np.concatenate([
                    frame_data['features'],
                    [
                        frame_data['attention_features']['central_focus'],
                        frame_data['attention_features']['gaze_fixation'],
                        frame_data['attention_features']['head_stability'],
                        frame_data['attention_features']['face_orientation'],
                        frame_data['attention_features']['attention_score']
                    ]
                ])
                frame_features.append(combined_features)
            
            # [30, 31] 형태의 시퀀스
            sequence_tensor = torch.FloatTensor(frame_features)
            processed_sequences.append(sequence_tensor)
            processed_labels.append(label)
        
        # 전체 데이터셋 저장
        dataset = {
            'sequences': processed_sequences,
            'labels': processed_labels,
            'feature_dim': 31,
            'sequence_length': 30,
            'num_classes': 2
        }
        
        torch.save(dataset, save_path)
        print(f"✅ 데이터셋 저장 완료: {len(sequences)}개 시퀀스")


def main():
    """ML → PyTorch 변환 실행"""
    print("🔄 ML 모델을 PyTorch로 변환합니다")
    
    # 기존 ML 모델 경로
    ml_model_path = input("ML 모델 경로 (Enter=기본값): ").strip() or \
                   "./xgboost_3class_concentration_classifier.pkl"
    
    if not os.path.exists(ml_model_path):
        print("❌ ML 모델 파일을 찾을 수 없습니다")
        return
    
    # 변환기 생성
    converter = MLToPyTorchConverter(ml_model_path)
    
    # 시퀀스 데이터 생성
    print("\n1️⃣ 30프레임 시퀀스 데이터 생성")
    num_sequences = int(input("생성할 시퀀스 수 (Enter=5000): ").strip() or "5000")
    sequences, labels = converter.simulate_30_frame_sequence(num_sequences)
    
    # PyTorch 데이터셋 저장
    print("\n2️⃣ PyTorch 데이터셋 저장")
    os.makedirs("./data", exist_ok=True)
    save_path = "./data/concentration_sequences.pt"
    converter.save_pytorch_dataset(sequences, labels, save_path)
    
    print(f"\n✅ 변환 완료!")
    print(f"📁 저장된 데이터셋: {save_path}")
    print(f"📊 총 시퀀스: {len(sequences)}개")
    print(f"📊 집중 비율: {sum(labels)/len(labels)*100:.1f}%")
    print(f"\n다음 단계: python train.py 로 PyTorch 모델 학습")

if __name__ == "__main__":
    main()
