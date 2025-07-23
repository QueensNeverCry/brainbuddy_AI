import cv2
import mediapipe as mp
"""
속도 향상을 위해 축소된 이미지에서 얼굴 검출 -> 원본 크기에 맞춰 bounding box 계산
"""

def crop_face(img_bgr, face_detector, fallback_to_full=True):
    h, w, _ = img_bgr.shape

    # ⏱️ 1. Resize for faster face detection
    scale = 0.25  # 이미지 크기 줄이기 (예: 640x480 → 160x120)
    resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rh, rw, _ = resized_rgb.shape

    # ⏱️ 2. Detect face on resized image
    results = face_detector.process(resized_rgb)

    if results.detections:
        max_area = 0
        best_bbox = None
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            area = bbox.width * bbox.height
            if area > max_area:
                max_area = area
                best_bbox = bbox

        if best_bbox:
            # ⏱️ 3. Rescale bbox to original image size
            x1 = max(int((best_bbox.xmin * rw) / scale), 0)
            y1 = max(int((best_bbox.ymin * rh) / scale), 0)
            x2 = min(x1 + int((best_bbox.width * rw) / scale), w)
            y2 = min(y1 + int((best_bbox.height * rh) / scale), h)
            face_crop = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            return face_rgb

    # 실패 시 전체 이미지 (RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if fallback_to_full else None

