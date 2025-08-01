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


# # 상태 변수 (전역 or 클래스 내부에서 관리)
# prev_bbox = None
# fail_count = 0
# MAX_FAIL = 10

# def crop_face(img_bgr, face_detector, fallback_to_full=True, margin_ratio=0.2):
#     global prev_bbox, fail_count
#     h, w, _ = img_bgr.shape
#     scale = 0.25

#     resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
#     resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#     rh, rw, _ = resized_rgb.shape

#     results = face_detector.process(resized_rgb)

#     def apply_margin(x1, y1, x2, y2, img_w, img_h, margin):
#         """좌표에 margin을 적용 (비율 기준)"""
#         w_box = x2 - x1
#         h_box = y2 - y1
#         mx = int(w_box * margin)
#         my = int(h_box * margin)
#         x1 = max(x1 - mx, 0)
#         y1 = max(y1 - my, 0)
#         x2 = min(x2 + mx, img_w)
#         y2 = min(y2 + my, img_h)
#         return x1, y1, x2, y2

#     if results.detections:
#         max_area = 0
#         best_bbox = None
#         for detection in results.detections:
#             bbox = detection.location_data.relative_bounding_box
#             area = bbox.width * bbox.height
#             if area > max_area:
#                 max_area = area
#                 best_bbox = bbox

#         if best_bbox:
#             # 성공 시: 이전 bbox 갱신 + 실패 카운터 초기화
#             prev_bbox = best_bbox
#             fail_count = 0

#             x1 = int((best_bbox.xmin * rw) / scale)
#             y1 = int((best_bbox.ymin * rh) / scale)
#             x2 = int(x1 + (best_bbox.width * rw) / scale)
#             y2 = int(y1 + (best_bbox.height * rh) / scale)
#             x1, y1, x2, y2 = apply_margin(x1, y1, x2, y2, w, h, margin_ratio)

#             face_crop = img_bgr[y1:y2, x1:x2]
#             return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

#     # 실패 시: 실패 카운터 증가
#     fail_count += 1

#     if prev_bbox and fail_count < MAX_FAIL:
#         # 이전 bbox 재사용
#         x1 = int((prev_bbox.xmin * rw) / scale)
#         y1 = int((prev_bbox.ymin * rh) / scale)
#         x2 = int(x1 + (prev_bbox.width * rw) / scale)
#         y2 = int(y1 + (prev_bbox.height * rh) / scale)
#         x1, y1, x2, y2 = apply_margin(x1, y1, x2, y2, w, h, margin_ratio)

#         face_crop = img_bgr[y1:y2, x1:x2]
#         return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

#     # 연속 실패가 10프레임 이상이면 스킵
#     if fail_count >= MAX_FAIL:
#         prev_bbox = None
#         return None  # 프레임 스킵

#     # 처음부터 실패 or fallback 옵션 사용
#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if fallback_to_full else None


