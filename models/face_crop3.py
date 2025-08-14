import cv2
import math
import os

# ✅ 전역 상태 저장
last_face_bbox = None  # (x1, y1, x2, y2)
miss_count = 0         # 연속 미검출 카운터
fail_save_index = 0    # 저장 파일 이름용 카운터

def _clamp_bbox_norm_to_pixels(bbox_norm, w, h,
                               margin_x=0.35, margin_y=0.15,
                               min_ratio=0.08):
    xmin, ymin, bw, bh = bbox_norm
    xmin = max(0.0, min(1.0, float(xmin)))
    ymin = max(0.0, min(1.0, float(ymin)))
    bw   = max(0.0, min(1.0, float(bw)))
    bh   = max(0.0, min(1.0, float(bh)))

    x1f = xmin * w
    y1f = ymin * h
    x2f = (xmin + bw) * w
    y2f = (ymin + bh) * h

    if margin_x > 0 or margin_y > 0:
        dx = (x2f - x1f) * margin_x
        dy = (y2f - y1f) * margin_y
        x1f -= dx; x2f += dx
        y1f -= dy; y2f += dy

    x1 = int(math.floor(max(0, x1f)))
    y1 = int(math.floor(max(0, y1f)))
    x2 = int(math.ceil (min(w, x2f)))
    y2 = int(math.ceil (min(h, y2f)))

    min_w = int(w * min_ratio); min_h = int(h * min_ratio)
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)
    if (x2 - x1) < min_w: x2 = min(w, x1 + min_w)
    if (y2 - y1) < min_h: y2 = min(h, y1 + min_h)

    return (x1, y1, x2, y2)

def _ema_smooth_bbox(prev_bbox, new_bbox, alpha=0.6):
    if prev_bbox is None:
        return new_bbox
    x1p, y1p, x2p, y2p = prev_bbox
    x1n, y1n, x2n, y2n = new_bbox
    x1 = int(alpha * x1n + (1 - alpha) * x1p)
    y1 = int(alpha * y1n + (1 - alpha) * y1p)
    x2 = int(alpha * x2n + (1 - alpha) * x2p)
    y2 = int(alpha * y2n + (1 - alpha) * y2p)
    return (x1, y1, x2, y2)

def crop_face(img_bgr, face_detector, fallback_to_full=True,
              scale=0.75, min_conf=0.6, miss_limit=10,
              margin_x=0.35, margin_y=0.15,
              min_box_ratio=0.08, ema_alpha=0.6,
              save_fail_dir=None):
    """
    - save_fail_dir: 검출 실패 시 이미지를 저장할 폴더 경로 (없으면 저장 안 함)
      저장 방식: 원본 / 이전 bbox 크롭 이미지 두 개 저장
    """
    global last_face_bbox, miss_count, fail_save_index

    h, w, _ = img_bgr.shape

    # 1) 다운스케일 & RGB 변환
    resized = cv2.resize(img_bgr, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 2) 얼굴 검출
    results = face_detector.process(resized_rgb)
    best = None
    best_area = 0.0

    if results.detections:
        for det in results.detections:
            if hasattr(det, "score") and det.score and det.score[0] < min_conf:
                continue
            rb = det.location_data.relative_bounding_box
            area = max(0.0, float(rb.width)) * max(0.0, float(rb.height))
            if area > best_area:
                best_area = area
                best = (rb.xmin, rb.ymin, rb.width, rb.height)

    if best is not None:
        bbox_px = _clamp_bbox_norm_to_pixels(
            best, w, h, margin_x=margin_x, margin_y=margin_y, min_ratio=min_box_ratio
        )
        bbox_px = _ema_smooth_bbox(last_face_bbox, bbox_px, alpha=ema_alpha)

        x1, y1, x2, y2 = _clamp_bbox_norm_to_pixels(
            ((bbox_px[0]/w), (bbox_px[1]/h),
             (bbox_px[2]-bbox_px[0])/w, (bbox_px[3]-bbox_px[1])/h),
            w, h, margin_x=0.0, margin_y=0.0, min_ratio=min_box_ratio
        )
        last_face_bbox = (x1, y1, x2, y2)
        miss_count = 0

        face_crop = img_bgr[y1:y2, x1:x2]
        return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # 3) 미검출 처리
    miss_count += 1
    if last_face_bbox is not None and miss_count <= miss_limit:
        x1, y1, x2, y2 = last_face_bbox
        face_crop = img_bgr[y1:y2, x1:x2]

        # 🔹 저장 기능 추가
        if save_fail_dir is not None:
            os.makedirs(save_fail_dir, exist_ok=True)
            fail_save_index += 1
            orig_path = os.path.join(save_fail_dir, f"fail_{fail_save_index:05d}_orig.jpg")
            crop_path = os.path.join(save_fail_dir, f"fail_{fail_save_index:05d}_crop.jpg")
            cv2.imwrite(orig_path, img_bgr)
            cv2.imwrite(crop_path, face_crop)

        return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    # 4) 장기 미검출 → 폴백
    if miss_count > miss_limit:
        last_face_bbox = None

        # 🔹 저장 기능 추가 (bbox 없음)
        if save_fail_dir is not None:
            os.makedirs(save_fail_dir, exist_ok=True)
            fail_save_index += 1
            orig_path = os.path.join(save_fail_dir, f"fail_{fail_save_index:05d}_orig.jpg")
            cv2.imwrite(orig_path, img_bgr)

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if fallback_to_full else None
