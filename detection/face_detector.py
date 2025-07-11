import mediapipe as mp
import cv2

#프레임에서 얼굴부분만 추출하는 함수
def extract_face(frame, face_mesh):
    # frame은 BGR 이미지
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))
        
        margin = 10
        x1, y1 = max(x1 - margin, 0), max(y1 - margin, 0)
        x2, y2 = min(x2 + margin, w), min(y2 + margin, h)
        
        # 얼굴 자를 때는 원본 BGR 이미지에서 자르기
        face_bgr = frame[y1:y2, x1:x2]
        
        if face_bgr.size == 0:
            return None
        
        # 자른 영역을 RGB로 변환 후 리사이즈
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        return face_resized
    
    return None


# 얼굴이 감지된 프레임들을 여러개 모아 리스트로 저장하는 함수
def detect_faces(frames,face_mesh):
    return [f for f in (extract_face(frame,face_mesh) for frame in frames) if f is not None]