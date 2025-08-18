import numpy as np
import math
from collections import deque
from typing import Dict, Tuple, Optional

class AttentionFeatureExtractor:
    """논문 기반 집중도 특징 추출기"""
    
    def __init__(self):
        # 직사각형 집중 영역 파라미터
        self.attention_zone_width = 720
        self.attention_zone_height = 180
        self.fixation_threshold = 4
        self.head_angle_threshold = 20
        self.stability_weight = 0.68
        
        # 시계열 데이터 추적
        self._gaze_history = deque(maxlen=10)
        self._fixation_frames = 0
        self._stability_score = 0.5
        
    def extract_features(self, frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Dict]:
        """특징 추출 (26차원 + attention 특징)"""
        # 논문 기반 attention 특징 계산
        attention_features = self.calculate_attention_features(face_box, frame.shape)
        
        # 26차원 특징 벡터 생성
        features = self.build_feature_vector(frame, face_box, attention_features)
        
        return features, attention_features
    
    def calculate_attention_features(self, face_box: Optional[Tuple[int, int, int, int]], 
                                   frame_shape: Tuple[int, int, int]) -> Dict:
        """논문 기반 집중도 특징 계산"""
        if face_box is None:
            return {
                'head_stability': 0.2,
                'gaze_fixation': 0.1,
                'central_focus': 0.0,
                'face_orientation': 0.0,
                'attention_score': 0.15
            }

        x, y, w, h = face_box
        cx, cy = x + w/2, y + h/2
        
        # 눈 위치 추정
        eye_center_x = x + w/2
        eye_center_y = y + h * 0.35
        
        screen_cx, screen_cy = frame_shape[1]//2, frame_shape[0]//2

        # 1. Central Focus Score (직사각형 영역 기반)
        rect_left = screen_cx - self.attention_zone_width // 2
        rect_right = screen_cx + self.attention_zone_width // 2
        rect_top = screen_cy - self.attention_zone_height // 2
        rect_bottom = screen_cy + self.attention_zone_height // 2
        
        if (rect_left <= eye_center_x <= rect_right and 
            rect_top <= eye_center_y <= rect_bottom):
            x_distance = abs(eye_center_x - screen_cx) / (self.attention_zone_width / 2)
            y_distance = abs(eye_center_y - screen_cy) / (self.attention_zone_height / 2)
            central_focus = max(0, 1 - max(x_distance, y_distance))
        else:
            central_focus = 0.0

        # 2. Head Orientation Score
        angle_deviation = abs(math.atan2(cy - screen_cy, cx - screen_cx) * 180 / math.pi)
        face_orientation = max(0, 1 - angle_deviation / self.head_angle_threshold)

        # 3. Gaze Fixation
        self._gaze_history.append((cx, cy))
        
        if len(self._gaze_history) >= 2:
            recent_movement = 0
            for i in range(1, min(3, len(self._gaze_history))):
                prev_x, prev_y = self._gaze_history[-i-1]
                curr_x, curr_y = self._gaze_history[-i]
                movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                recent_movement += movement

            if recent_movement < 50:
                self._fixation_frames += 1
            else:
                self._fixation_frames = max(0, self._fixation_frames - 1)

            gaze_fixation = min(1.0, self._fixation_frames / self.fixation_threshold)
        else:
            gaze_fixation = 0.0

        # 4. Head Stability
        face_size_consistency = min(1.0, (w * h) / 15000)
        head_stability = (face_orientation + face_size_consistency) / 2

        # 5. 종합 Attention Score
        attention_score = (
            central_focus * 0.4 +
            gaze_fixation * 0.3 +
            head_stability * self.stability_weight * 0.2 +
            face_orientation * 0.1
        )

        # 안정적인 스무딩
        self._stability_score = 0.7 * self._stability_score + 0.3 * attention_score

        return {
            'head_stability': head_stability,
            'gaze_fixation': gaze_fixation,
            'central_focus': central_focus,
            'face_orientation': face_orientation,
            'attention_score': self._stability_score
        }
    
    def build_feature_vector(self, frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]], 
                           attention_features: Dict) -> np.ndarray:
        """26차원 특징 벡터 생성"""
        vec = np.zeros(26, dtype=np.float32)
        
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

        return vec
