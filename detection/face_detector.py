import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0)

def extract_face(frame):
    results = face_detector.process(frame)
    if results.detections:
        box = results.detections[0].location_data.relative_bounding_box
        h, w, _ = frame.shape
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        x2 = x1 + int(box.width * w)
        y2 = y1 + int(box.height * h)
        face = frame[y1:y2, x1:x2]
        
        if face.size ==0:
            return None
        
        return cv2.resize(face, (224, 224))
    return None

def detect_faces(frames):
    return [f for f in (extract_face(frame) for frame in frames) if f is not None]
