import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection

def crop_face(img_bgr, fallback_to_full=True):
    """
    BGR 이미지에서 얼굴만 crop해서 반환. 얼굴 없을 경우 원본 전체 반환 가능.
    Args:
        img_bgr (np.array): OpenCV 이미지 (BGR)
        fallback_to_full (bool): 얼굴 검출 실패 시 원본 반환 여부
    Returns:
        face_crop (np.array): 얼굴 영역 (RGB)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
        results = face_detector.process(img_rgb)

    if results.detections:
        # 여러 얼굴 중 가장 큰 얼굴 선택 (넓이 기준)
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

    # 얼굴이 하나도 없을 경우
    if fallback_to_full:
        return img_rgb
    else:
        return None