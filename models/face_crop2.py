# models/face_crop.py
import cv2
from typing import Optional, Tuple

# bbox 형식: (x1, y1, x2, y2) in original-frame coords

def _clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def crop_face(
    img_bgr,
    face_detector,                       # mp.solutions.face_detection.FaceDetection
    fallback_to_full: bool = True,
    prev_bbox: Optional[Tuple[int,int,int,int]] = None,
    scale: float = 0.5,                  # 탐지 속도/성능 트레이드오프 (0.25~1.0)
    margin: float = 0.12                 # bbox 주변 여유 비율
):
    """
    얼굴을 탐지해 RGB crop과 bbox를 반환.
    탐지 실패 시 prev_bbox가 있으면 그것으로 crop.
    둘 다 없으면 (전체 RGB, None) 또는 (None, None) 반환 (fallback_to_full에 따름).
    return: (face_rgb, bbox)  # bbox는 (x1,y1,x2,y2) 또는 None
    """
    h, w = img_bgr.shape[:2]

    # 1) 다운스케일 이미지로 탐지
    if scale != 1.0:
        resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        resized = img_bgr
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 2) 탐지 실행
    results = face_detector.process(resized_rgb)

    best = None
    if results.detections:
        max_area = 0.0
        for det in results.detections:
            rb = det.location_data.relative_bounding_box  # (xmin,ymin,width,height) in [0,1] of resized
            area = rb.width * rb.height
            if area > max_area:
                max_area = area
                best = rb

    if best is not None:
        # 3) 상대 좌표 → 원본 프레임 좌표
        x1 = int(best.xmin * w)
        y1 = int(best.ymin * h)
        bw = int(best.width * w)
        bh = int(best.height * h)
        x2 = x1 + bw
        y2 = y1 + bh

        # 4) 마진 적용
        dx = int(bw * margin)
        dy = int(bh * margin)
        x1 -= dx; y1 -= dy; x2 += dx; y2 += dy

        bbox = _clip_bbox(x1, y1, x2, y2, w, h)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_crop = img_bgr[y1:y2, x1:x2]
            return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), bbox

    # 5) 탐지 실패 시 prev_bbox 사용
    if prev_bbox is not None:
        x1, y1, x2, y2 = prev_bbox
        bbox = _clip_bbox(x1, y1, x2, y2, w, h)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face_crop = img_bgr[y1:y2, x1:x2]
            return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), bbox

    # 6) 최후: 전체 프레임 또는 None
    if fallback_to_full:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), None
    return None, None
