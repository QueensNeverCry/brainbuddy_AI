from ultralytics import YOLO
import numpy as np
import cv2
import os

# YOLOv8 얼굴 모델 로딩 (전역)
model_path = os.path.join(os.path.dirname(__file__), "yolov8n-face-lindevs.pt")
yolo_model = YOLO(model_path)

def crop_face_batch(img_list, conf=0.4, iou=0.5):
    """
    여러 이미지를 YOLO로 배치 처리하여 얼굴 crop을 반환.
    얼굴이 감지되지 않으면 원본 이미지를 반환.
    :param img_list: List[np.ndarray]
    :return: List[np.ndarray]
    """
    results = yolo_model.predict(img_list, conf=conf, iou=iou, verbose=False)
    cropped_faces = []

    for img, result in zip(img_list, results):
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []

        if len(boxes) == 0:
            # 얼굴이 감지되지 않으면 원본 이미지 반환
            cropped_faces.append(img)
            continue

        # 가장 큰 얼굴 선택
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
        largest_idx = areas.index(max(areas))
        x1, y1, x2, y2 = boxes[largest_idx]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

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

def crop_face_batch_chunked(img_list, batch_size):
    """
    YOLO 얼굴 검출을 메모리 초과 없이 chunk 단위로 나눠 처리
    :param img_list: List[np.ndarray]
    :param batch_size: YOLO 처리 배치 크기
    :return: List[np.ndarray]
    """
    results = []
    for i in range(0, len(img_list), batch_size):
        chunk = img_list[i:i + batch_size]
        chunk_results = crop_face_batch(chunk)  # 기존 배치 처리 함수
        results.extend(chunk_results)
    return results