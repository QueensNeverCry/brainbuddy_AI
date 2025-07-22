import cv2
import mediapipe as mp


# 수정된 crop_face 함수
def crop_face(img_bgr, face_detector, fallback_to_full=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    results = face_detector.process(img_rgb)

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
            x1 = max(int(best_bbox.xmin * w), 0)
            y1 = max(int(best_bbox.ymin * h), 0)
            x2 = min(x1 + int(best_bbox.width * w), w)
            y2 = min(y1 + int(best_bbox.height * h), h)
            face_crop = img_rgb[y1:y2, x1:x2]
            return face_crop

    return img_rgb if fallback_to_full else None # 얼굴 검출이 안되면 전체 이미지를 반환
