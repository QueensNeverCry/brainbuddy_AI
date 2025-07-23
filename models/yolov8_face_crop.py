from ultralytics import YOLO
import cv2

# YOLOv8 얼굴탐지 모델 로드 (가중치 경로는 상대/절대 경로 가능)
model = YOLO("yolov8s-face-lindevs.pt")  # GPU 자동 사용

def crop_face_with_yolo(image):
    results = model.predict(image, imgsz=640, conf=0.5, verbose=False)
    
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    if len(boxes) == 0:
        return image  # 얼굴 없으면 원본 이미지 반환

    x1, y1, x2, y2 = boxes[0]
    face_crop = image[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return image
    return face_crop
