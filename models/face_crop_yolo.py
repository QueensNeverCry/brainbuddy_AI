# face_crop_yolo.py

import os
import cv2
import numpy as np
from ultralytics import YOLO

# ✅ 모델을 전역에서 한 번만 로드
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n-face-lindevs.pt")
yolo_model = YOLO(MODEL_PATH)

def crop_face_batch_chunked(img_list, batch_size=2, conf=0.4, iou=0.5):
    """
    이미지를 chunk로 나눠 YOLO로 얼굴을 검출하고 crop된 얼굴 리스트를 반환.
    검출 실패 시 원본 이미지를 반환.
    """
    cropped_faces = []
    
    for i in range(0, len(img_list), batch_size):
        chunk = img_list[i:i+batch_size]
        results = yolo_model.predict(chunk, conf=conf, iou=iou, verbose=False)

        for img, result in zip(chunk, results):
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []

            if len(boxes) == 0:
                cropped_faces.append(img)  # 검출 실패 → 원본
                continue

            # 가장 큰 얼굴 선택
            areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
            largest_idx = areas.index(max(areas))
            x1, y1, x2, y2 = map(int, boxes[largest_idx])
            face = img[y1:y2, x1:x2]

            # 정사각형으로 패딩
            h, w = face.shape[:2]
            size = max(h, w)
            pad_top = (size - h) // 2
            pad_bottom = size - h - pad_top
            pad_left = (size - w) // 2
            pad_right = size - w - pad_left

            square_face = cv2.copyMakeBorder(
                face, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            cropped_faces.append(square_face)

    return cropped_faces
