import cv2
import numpy as np
from ml_classifier import ConcentrationClassifier
import time, os
from collections import deque, Counter
import math


class ConcentrationInference:
    """논문 기반 실시간 집중도 분석 시스템 (30프레임당 0/1 출력)"""

    def __init__(self, model_path: str):
        # 모델 로드
        self.classifier = ConcentrationClassifier()
        self.classifier.load_model(model_path)

        # 🔥 2클래스 정의로 변경
        self.cls_name = {0: 'Not Focused', 1: 'Focused'}
        self.cls_color = {0: (0, 0, 255), 1: (0, 255, 0)}  # 빨강, 초록

        # 얼굴 검출 및 안정화
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_face_box = None
        self.face_lost_count = 0
        self.face_keep_frames = 10  # 30프레임 간격에 맞게 증가

        # 🔥 30프레임 간격에 최적화된 예측 안정화
        self.pred_buffer = deque(maxlen=2)  # 2개 결과만 보관

        # 🔥 직사각형 집중 영역 파라미터로 변경
        self.attention_zone_width = 720    # 가로 720픽셀 (기존 원 지름 360의 2배)
        self.attention_zone_height = 180   # 세로 180픽셀 (기존 원 반지름과 동일)
        self.fixation_threshold = 4         
        self.head_angle_threshold = 20      
        self.stability_weight = 0.68

        # 🔥 집중 탐지 임계값들 (균형잡힌 조정)
        self.focus_sensitivity = {
            'central_focus_threshold': 0.7,     # 중앙 응시 40%
            'gaze_fixation_threshold': 0.7,     # 고정 응시 60%
            'attention_score_threshold': 0.5,   # 종합 점수 50%
            'model_confidence_threshold': 0.25, # 모델 확률 25%
            'focus_indicator_threshold': 3      # 7점 중 3점
        }

        # 시계열 데이터 추적 (30프레임 간격에 맞게 조정)
        self._gaze_history = deque(maxlen=10)  # 15 → 10으로 조정
        self._fixation_frames = 0
        self._stability_score = 0.5

        # 🔥 30프레임 간격용 로깅
        self.last_log_t = 0
        self.log_interval = 1.0  # 1초마다 로깅

        # 🔥 30프레임 분석 결과 저장
        self.analysis_results = []

        print("✅ 30프레임 간격 집중도 분석기 초기화 완료")
        print("📚 백엔드 10초 데이터 최적화")
        print("🎯 30프레임당 0/1 출력 모드")

    def detect_face(self, frame):
        """얼굴 검출 (오탐지 방지 강화)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # 🔥 오탐지 방지를 위한 엄격한 설정
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.08,     # 더 엄격하게
            minNeighbors=6,       # 더 많은 이웃 필요
            minSize=(100, 100),   # 더 큰 최소 크기
            maxSize=(350, 350),   # 더 작은 최대 크기
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces):
            largest_face = max(faces, key=lambda b: b[2]*b[3])
            x, y, w, h = largest_face
            
            # 🔥 간단한 검증: 화면 중앙 근처에 있고, 적정 크기인가?
            frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
            face_center_x, face_center_y = x + w//2, y + h//2
            
            distance = np.sqrt((face_center_x - frame_center_x)**2 + 
                              (face_center_y - frame_center_y)**2)
            
            # 화면 중앙에서 400픽셀 이내 + 적정 크기일 때만 얼굴로 인정
            if distance < 400 and 100 <= w <= 350 and 100 <= h <= 350:
                self.last_face_box = largest_face
                self.face_lost_count = 0
                return self.last_face_box, True
            else:
                print("⚠️ 배경 오탐지 방지: 얼굴이 아닌 것으로 판단")
        
        # 기존 얼굴 추적 로직 (30프레임 간격에 맞게 더 오래 유지)
        self.face_lost_count += 1
        if self.last_face_box is not None and self.face_lost_count < self.face_keep_frames:
            return self.last_face_box, False
        self.last_face_box = None
        return None, False

    def calculate_attention_features(self, face_box, frame_shape):
        """논문 기반 집중도 특징 계산 (직사각형 집중 영역 기반)"""
        if face_box is None:
            return {
                'head_stability': 0.2,
                'gaze_fixation': 0.1,
                'central_focus': 0.0,
                'face_orientation': 0.0,
                'attention_score': 0.15
            }

        x, y, w, h = face_box
        
        # 기존: 얼굴 중심점 사용
        cx, cy = x + w/2, y + h/2  # 고정 응시 등 기존 계산용
        
        # 눈 위치 추정 (해부학적 비율 사용)
        eye_center_x = x + w/2                # 얼굴 중앙 X (좌우 눈의 중점)
        eye_center_y = y + h * 0.35          # 얼굴 상단에서 35% 지점 (눈 높이)
        
        screen_cx, screen_cy = frame_shape[1]//2, frame_shape[0]//2

        # 1. 🔥 Central Focus Score (직사각형 영역 기반으로 수정)
        # 화면 중앙의 직사각형 영역 정의
        rect_left = screen_cx - self.attention_zone_width // 2
        rect_right = screen_cx + self.attention_zone_width // 2
        rect_top = screen_cy - self.attention_zone_height // 2
        rect_bottom = screen_cy + self.attention_zone_height // 2
        
        # 눈 위치가 직사각형 안에 있는지 확인
        if (rect_left <= eye_center_x <= rect_right and 
            rect_top <= eye_center_y <= rect_bottom):
            # 직사각형 안에 있으면 중심에서의 거리에 따라 점수 계산
            # X축 거리 (가로 방향)
            x_distance = abs(eye_center_x - screen_cx) / (self.attention_zone_width / 2)
            # Y축 거리 (세로 방향) 
            y_distance = abs(eye_center_y - screen_cy) / (self.attention_zone_height / 2)
            
            # 직사각형 중심에서 멀수록 점수 감소
            central_focus = max(0, 1 - max(x_distance, y_distance))
        else:
            # 직사각형 밖에 있으면 0점
            central_focus = 0.0

        # 2. Head Orientation Score (기존 얼굴 중심 기준 유지)
        angle_deviation = abs(math.atan2(cy - screen_cy, cx - screen_cx) * 180 / math.pi)
        face_orientation = max(0, 1 - angle_deviation / self.head_angle_threshold)

        # 3. Gaze Fixation (기존 얼굴 중심 기준 유지)
        self._gaze_history.append((cx, cy))
        
        if len(self._gaze_history) >= 2:
            recent_movement = 0
            for i in range(1, min(3, len(self._gaze_history))):
                prev_x, prev_y = self._gaze_history[-i-1]
                curr_x, curr_y = self._gaze_history[-i]
                movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                recent_movement += movement

            # 고정 응시 판단
            if recent_movement < 50:
                self._fixation_frames += 1
            else:
                self._fixation_frames = max(0, self._fixation_frames - 1)

            gaze_fixation = min(1.0, self._fixation_frames / self.fixation_threshold)
        else:
            gaze_fixation = 0.0

        # 4. Head Stability (기존 방식 유지)
        face_size_consistency = min(1.0, (w * h) / 15000)
        head_stability = (face_orientation + face_size_consistency) / 2

        # 5. 직사각형 기반 정밀 집중도 추가 검증
        if central_focus > 0.8:  # 직사각형 중앙에 가까울 때
            rectangular_precision_bonus = 0.1
        else:
            rectangular_precision_bonus = 0.0

        # 6. 종합 Attention Score (직사각형 기반 central_focus 반영)
        attention_score = (
            central_focus * 0.4 +                    # 🔥 직사각형 영역 기반 중앙 집중
            gaze_fixation * 0.3 +                    
            head_stability * self.stability_weight * 0.2 +
            face_orientation * 0.1 +                 
            rectangular_precision_bonus              # 🔥 직사각형 정밀도 보너스
        )

        # 30프레임 간격에 맞는 안정적인 스무딩
        self._stability_score = 0.7 * self._stability_score + 0.3 * attention_score

        return {
            'head_stability': head_stability,
            'gaze_fixation': gaze_fixation,
            'central_focus': central_focus,          # 🔥 이제 직사각형 영역 기반 값
            'face_orientation': face_orientation,
            'attention_score': self._stability_score
        }



    def build_research_based_features(self, frame, face_box):
        """연구 기반 특징 벡터 생성"""
        vec = np.zeros(26, dtype=np.float32)
        attention_features = self.calculate_attention_features(face_box, frame.shape)

        if face_box is not None:
            x, y, w, h = face_box
            cx, cy = x + w/2, y + h/2

            attention_score = attention_features['attention_score']
            
            if attention_score > 0.5:
                # 고집중 특징
                vec[0:3] = [0.0, 0.0, 0.0]
                vec[4] = 640; vec[5] = 360
                vec[13:15] = [0.5, 0.5]
                vec[15] = 0.95; vec[16] = 0.9
                vec[17] = 5.0; vec[18] = 0.02
                vec[19] = min(20, self._fixation_frames)
                vec[21] = attention_features['central_focus']
                
            elif attention_score > 0.25:
                # 보통집중 특징
                vec[0:3] = [0.5, 0.5, 0.2]
                vec[4] = cx; vec[5] = cy
                vec[13:15] = [1.5, 1.5]
                vec[15] = 0.8; vec[16] = 0.8
                vec[17] = 15.0; vec[18] = 0.05
                vec[19] = min(15, self._fixation_frames)
                vec[21] = attention_features['central_focus'] * 0.8
                
            else:
                # 저집중 특징
                vec[0:3] = [3.0, 2.5, 2.0]
                vec[4] = cx + np.random.normal(0, 50)
                vec[5] = cy + np.random.normal(0, 50)
                vec[13:15] = [4.0, 3.5]
                vec[15] = 0.3; vec[16] = 0.4
                vec[17] = 50.0; vec[18] = 0.4
                vec[19] = max(2, self._fixation_frames)
                vec[21] = attention_features['central_focus'] * 0.3

            # 공통 특징
            vec[3] = min(100, 90000 / max(w*h, 1000) + 45)
            vec[6:10] = [cx-20, cy-10, cx+20, cy-10]
            vec[10:12] = [0.3, 0.3]
            vec[12] = abs((cy - 360) / 360) * 5
            vec[20] = attention_features['gaze_fixation'] * 10
            vec[22] = min(0.8, attention_features['head_stability'])

        return vec, attention_features

    def focus_sensitive_prediction(self, feat_vec, attention_features):
        """🔥 30프레임 최적화된 집중 탐지 예측"""
        
        # 1단계: 기본 3클래스 모델 예측
        raw_pred, probs = self.classifier.predict(feat_vec.reshape(1, -1))
        raw_cls = raw_pred[0]
        
        # 2단계: 집중 지표 점수 계산
        focus_indicators = 0
        sensitivity = self.focus_sensitivity
        
        # 🔥 원본 모델 예측에 훨씬 더 높은 가중치
        if raw_cls == 2:  # 집중 예측
            focus_indicators += 4  # 기존 3 → 4
        elif raw_cls == 0:  # 🔥 비집중 예측 시 강한 페널티 추가
            focus_indicators -= 4  # 새로 추가: 비집중이면 -4점
        elif probs[0][2] > sensitivity['model_confidence_threshold']:
            focus_indicators += 2
        
        # 🔥 원본 모델이 확신할 때 추가 페널티/보너스
        if probs[0][0] > 0.8:  # 비집중 확률 80% 이상
            focus_indicators -= 3  # 추가 페널티
        elif probs[0][2] > 0.8:  # 집중 확률 80% 이상
            focus_indicators += 3  # 추가 보너스
        
        # 논문 기반 지표들
        if attention_features['central_focus'] > sensitivity['central_focus_threshold']:
            focus_indicators += 2
            
        if attention_features['gaze_fixation'] > sensitivity['gaze_fixation_threshold']:
            focus_indicators += 2
            
        if attention_features['attention_score'] > sensitivity['attention_score_threshold']:
            focus_indicators += 2
        
        # 추가 조건들
        if attention_features['head_stability'] > 0.4:
            focus_indicators += 1
            
        if attention_features['face_orientation'] > 0.3:
            focus_indicators += 1
        
        # 3단계: 집중 판정
        if focus_indicators >= sensitivity['focus_indicator_threshold']:
            binary_result = 1  # 집중
            confidence = min(0.95, 0.6 + 
                           attention_features['attention_score'] * 0.2 + 
                           probs[0][2] * 0.15)
        else:
            binary_result = 0  # 집중안함
            confidence = max(0.5, (probs[0][0] + probs[0][1]) / 2)
        
        # 🔥 4단계: 30프레임 간격용 시간적 안정화
        self.pred_buffer.append(binary_result)
        
        if len(self.pred_buffer) >= 2:
            # 집중 우호적 다수결 (2개 중 1개만 집중이어도 집중으로)
            focus_count = sum(1 for x in self.pred_buffer if x == 1)
            if focus_count >= 1:
                final_result = 1
            else:
                final_result = 0
        else:
            final_result = binary_result
        
        return raw_cls, final_result, probs[0], confidence

    def log_detailed_analysis(self, frame_idx, face_status, attention_features, raw_cls, final_cls, conf, probs):
        """30프레임 간격용 상세 로그"""
        now = time.time()
        if now - self.last_log_t < self.log_interval:
            return
        self.last_log_t = now

        if face_status == 'miss':
            print(f"[Frame {frame_idx:4d}] ❌ 얼굴 없음 → 결과: 0 (비집중)")
            return

        status_icon = "🎯" if face_status == 'detect' else "📍"
        raw_name = "Focused" if raw_cls == 2 else "Not Focused" if raw_cls == 0 else "Distracted"
        final_name = self.cls_name[final_cls] if final_cls is not None else "None"
        
        # 🔥 30프레임 결과 강조 로깅
        print(f"\n{'='*80}")
        print(f"[Frame {frame_idx:4d}] 📊 30프레임 분석 결과: {final_cls} ({'집중' if final_cls == 1 else '비집중'})")
        print(f"{'='*80}")
        print(f"{status_icon} Raw: {raw_name:12s} → Final: {final_name:12s} (확신도: {conf:.3f})")
        print(f"📊 원본 확률: [ 비집중:{probs[0]:.2f}  주의산만:{probs[1]:.2f}  집중:{probs[2]:.2f} ]")
        
        # 논문 기반 분석 지표
        att = attention_features
        print(f"🎯 지표 분석:")
        print(f"   - 종합 점수: {att['attention_score']:.2f}")
        print(f"   - 중앙 집중: {att['central_focus']:.2f}")
        print(f"   - 고정 응시: {att['gaze_fixation']:.2f}")
        print(f"   - 머리 안정성: {att['head_stability']:.2f}")
        print(f"{'='*80}\n")

    def draw_binary_ui(self, frame, face_box, face_status, binary_cls, conf, attention_features):
        """30프레임 분석용 UI (직사각형 집중 영역 표시)"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (550, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 얼굴 박스
        if face_box is not None:
            x, y, w, h = face_box
            
            if binary_cls == 1:
                box_color = (0, 255, 0)  # 초록: 집중
            else:
                box_color = (0, 0, 255)  # 빨강: 집중안함
                
            cv2.rectangle(frame, (x-3, y-3), (x+w+3, y+h+3), box_color, 3)
            
            # 🔥 직사각형 집중 영역 표시 (기존 원 대신)
            center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
            
            # 직사각형 좌표 계산
            rect_left = center_x - self.attention_zone_width // 2
            rect_right = center_x + self.attention_zone_width // 2
            rect_top = center_y - self.attention_zone_height // 2
            rect_bottom = center_y + self.attention_zone_height // 2
            
            # 직사각형 그리기 (흰색 테두리)
            cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (255, 255, 255), 2)

        # 30프레임 분석 결과 표시
        if binary_cls is not None:
            state_text = "FOCUSED" if binary_cls == 1 else "NOT FOCUSED"
            color = self.cls_color[binary_cls]
            
            cv2.putText(frame, f"State: {state_text}", (40, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {conf:.3f}", (40, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 논문 기반 지표
        att = attention_features
        cv2.putText(frame, f"Attention Score: {att['attention_score']:.2f}", (40, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Central Focus: {att['central_focus']:.2f}", (40, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Gaze Fixation: {att['gaze_fixation']:.2f}", (40, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 30프레임 분석 모드 표시
        cv2.putText(frame, "Mode: 30-Frame Analysis (Rectangular Focus Zone)", (40, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 255, 128), 1)
        cv2.putText(frame, f"Focus Area: {self.attention_zone_width}x{self.attention_zone_height} pixels", (40, 260), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 128), 1)
        cv2.putText(frame, "Research: Zhang(2019), Duchowski(2018), Kim(2020)", (40, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

        return frame


    def run(self):
        """30프레임 간격 메인 실행 루프"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다")
            return

        # 초기화
        self._gaze_history = deque(maxlen=10)
        self._fixation_frames = 0

        f_idx, proc_cnt = 0, 0
        t0 = time.time()

        # 상태 변수 초기화
        face_status = 'miss'
        binary_cls = None
        conf = 0.0
        attention_features = {
            'attention_score': 0.0, 
            'central_focus': 0.0,
            'gaze_fixation': 0.0, 
            'head_stability': 0.0
        }

        print("🚀 30프레임 간격 집중도 분석 시작")
        print("📚 백엔드 10초 데이터에 최적화")
        print("🎯 30프레임마다 0/1 출력")
        print("⏱️  초당 1회 분석 (30fps → 1fps)")
        print("ESC/Q로 종료\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            f_idx += 1

            # 🔥 30프레임마다 처리 (초당 1회)
            if f_idx % 30 == 0:
                proc_cnt += 1
                face_detection_result = self.detect_face(frame)
                
                if face_detection_result[0] is not None:
                    face_box, is_detect = face_detection_result
                else:
                    face_box, is_detect = None, False
                
                if is_detect:
                    face_status = 'detect'
                elif face_box is not None:
                    face_status = 'track'
                else:
                    face_status = 'miss'

                if face_box is not None:
                    try:
                        feat_vec, attention_features = self.build_research_based_features(frame, face_box)
                        raw_cls, binary_cls, probs, conf = self.focus_sensitive_prediction(feat_vec, attention_features)
                        
                        # 🔥 30프레임 결과 저장 및 로깅
                        result = {
                            'frame': f_idx,
                            'timestamp': time.time(),
                            'result': binary_cls,
                            'confidence': conf
                        }
                        self.analysis_results.append(result)
                        
                        self.log_detailed_analysis(f_idx, face_status, attention_features, raw_cls, binary_cls, conf, probs)
                        
                    except Exception as e:
                        print(f"❌ 예측 오류: {e}")
                        binary_cls = 0  # 오류 시 비집중
                        conf = 0.0
                        
                        # 오류 결과도 저장
                        result = {
                            'frame': f_idx,
                            'timestamp': time.time(),
                            'result': 0,
                            'confidence': 0.0
                        }
                        self.analysis_results.append(result)
                else:
                    binary_cls = 0  # 얼굴 없음 시 비집중
                    conf = 0.0
                    
                    # 얼굴 없음 결과 저장
                    result = {
                        'frame': f_idx,
                        'timestamp': time.time(),
                        'result': 0,
                        'confidence': 0.0
                    }
                    self.analysis_results.append(result)
                    
                    print(f"\n[Frame {f_idx:4d}] ❌ 얼굴 없음 → 결과: 0 (비집중)\n")

            # UI 업데이트 (매 프레임)
            current_face_box = self.last_face_box
            frame = self.draw_binary_ui(frame, current_face_box, face_status, binary_cls, conf, attention_features)

            # 🔥 30프레임 진행 상황 표시
            elapsed = time.time() - t0
            fps = f_idx / elapsed if elapsed > 0 else 0
            next_analysis = 30 - (f_idx % 30)
            
            cv2.putText(frame, f"FPS: {fps:4.1f} | Analysis: {proc_cnt}", (1020, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Next in: {next_analysis} frames", (1020, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("30-Frame Concentration Analysis", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

        # 🔥 종료 시 결과 요약
        dur = time.time() - t0
        print(f"\n{'='*80}")
        print(f"📊 30프레임 분석 완료")
        print(f"{'='*80}")
        print(f"총 프레임: {f_idx} | 분석 횟수: {proc_cnt} | 평균 FPS: {f_idx/dur:.1f}")
        print(f"분석 간격: 30프레임 (초당 1회)")
        print(f"총 분석 결과: {len(self.analysis_results)}개")
        
        if self.analysis_results:
            focus_count = sum(1 for r in self.analysis_results if r['result'] == 1)
            focus_ratio = focus_count / len(self.analysis_results) * 100
            print(f"집중 판정: {focus_count}/{len(self.analysis_results)} ({focus_ratio:.1f}%)")
        
        print(f"{'='*80}")
        
        cap.release()
        cv2.destroyAllWindows()

    def get_latest_result(self):
        """백엔드 연동용: 최신 분석 결과 반환"""
        if self.analysis_results:
            return self.analysis_results[-1]
        return None

    def get_results_summary(self, last_n=10):
        """백엔드 연동용: 최근 N개 결과 요약"""
        if not self.analysis_results:
            return None
            
        recent_results = self.analysis_results[-last_n:]
        focus_count = sum(1 for r in recent_results if r['result'] == 1)
        
        return {
            'total_analyzed': len(recent_results),
            'focus_count': focus_count,
            'focus_ratio': focus_count / len(recent_results) if recent_results else 0,
            'latest_result': recent_results[-1] if recent_results else None
        }


def main():
    model_path = input("모델 경로 (Enter=기본값): ").strip() or \
                 "./json_features_3class_concentration_classifier.pkl"
    
    if not os.path.exists(model_path):
        print("❌ 모델 파일 없음")
        return
    
    try:
        ConcentrationInference(model_path).run()
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
