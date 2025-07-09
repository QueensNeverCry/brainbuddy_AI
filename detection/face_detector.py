import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True) #Fesh Mesh 모델 초기화

#프레임에서 얼굴부분만 추출하는 함수
def extract_face(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# openCV는 이미지를 BGR형식으로 불러옴 -> 이를 RGB 순서로 변환(다른 라이브러리나 딥러닝 모델은 RGB를 주로 사용하기 때문)
    h,w,_ =frame.shape # 가로,세로,채널수
    results =face_mesh.process(frame) #frame에서 얼굴의 랜드마크를 추출
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # 전체 랜드마크에서 얼굴 박스를 추정하기 위한 좌표 리스트
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        
        #얼굴 둘러싸는 최소 직사각형 좌표 계산
        x1, y1 = int(min(x_coords)), int(min(y_coords))
        x2, y2 = int(max(x_coords)), int(max(y_coords))

        #얼굴 주위로 여유 공간을 주기 위해 마진 추가
        margin = 10
        x1, y1 = max(x1 - margin, 0), max(y1 - margin, 0)
        x2, y2 = min(x2 + margin, w), min(y2 + margin, h)
        
        #얼굴 영역만 잘라내기
        face = frame[y1:y2, x1:x2]

        #얼굴이 없는 경우
        if face.size == 0:
            return None

        return cv2.resize(face, (224, 224))#CNN 이미지 크기로 resize
    return None

# 얼굴이 감지된 프레임들을 여러개 모아 리스트로 저장하는 함수
def detect_faces(frames):
    return [f for f in (extract_face(frame) for frame in frames) if f is not None]