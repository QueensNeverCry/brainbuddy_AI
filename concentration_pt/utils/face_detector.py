import cv2
import numpy as np
from typing import Optional, Tuple

class FaceDetector:
    """얼굴 검출기"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.last_face_box = None
        self.face_lost_count = 0
        self.face_keep_frames = 10
        
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """얼굴 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=6,
            minSize=(100, 100),
            maxSize=(350, 350),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            largest_face = max(faces, key=lambda b: b[2] * b[3])
            
            # 검증
            if self._validate_face(largest_face, frame.shape):
                self.last_face_box = largest_face
                self.face_lost_count = 0
                return tuple(largest_face)
        
        # 얼굴 추적
        self.face_lost_count += 1
        if self.last_face_box is not None and self.face_lost_count < self.face_keep_frames:
            return tuple(self.last_face_box)
        
        self.last_face_box = None
        return None
    
    def _validate_face(self, face_box: Tuple[int, int, int, int], 
                      frame_shape: Tuple[int, int, int]) -> bool:
        """얼굴 검출 결과 검증"""
        x, y, w, h = face_box
        frame_h, frame_w = frame_shape[:2]
        
        # 화면 중앙 거리 확인
        frame_center_x, frame_center_y = frame_w // 2, frame_h // 2
        face_center_x, face_center_y = x + w // 2, y + h // 2
        
        distance = np.sqrt((face_center_x - frame_center_x)**2 + 
                          (face_center_y - frame_center_y)**2)
        
        return distance < 400 and 100 <= w <= 350 and 100 <= h <= 350
