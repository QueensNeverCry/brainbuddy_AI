import mediapipe as mp
import cv2

def extract_face(frame, face_mesh, debug=False):
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
        
        face_bgr = frame[y1:y2, x1:x2]
        
        if face_bgr.size == 0:
            return None
        
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        if debug:
            # 디버깅용: 추출된 얼굴 이미지 출력
            cv2.imshow("Extracted Face", cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)  # 아무 키나 누르면 다음으로 진행
            cv2.destroyAllWindows()
        
        return face_resized
    
    else:
        if debug:
            print("얼굴을 찾지 못했습니다.")
    
    return None
