import os
import cv2
import mediapipe as mp
import numpy as np
import platform
import re

def windows_path_to_wsl(path: str) -> str:
    """
    WSL 환경일 때 'C:\\...' 형태의 경로를 '/mnt/c/...' 형태로 변환.
    Windows 네이티브 Python이면 그대로 반환.
    """
    if platform.system() == "Linux" and re.match(r"^[A-Za-z]:\\", path):
        drive, rest = path.split(":", 1)
        rest = rest.replace("\\", "/")
        return f"/mnt/{drive.lower()}{rest}"
    return path

def robust_imread(path: str):
    """
    1) cv2.imread 로 시도
    2) Windows UNC 경로(\\\\?\\)로 재시도
    3) open + cv2.imdecode 로 바이트 디코딩 우회
    """
    # 1) 일반 읽기 시도
    img = cv2.imread(path)
    if img is not None:
        return img

    # 2) Windows에서 UNC 경로 재시도
    if os.name == "nt":
        abs_path = os.path.abspath(path)
        unc_path = r"\\\\?\\" + abs_path
        img = cv2.imread(unc_path)
        if img is not None:
            return img

    # 3) 파일 바이트 읽어 imdecode 우회
    try:
        with open(path, "rb") as f:
            data = f.read()
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def crop_face(img_bgr, detector, fallback_to_full=True):
    """
    축소된 이미지에서 가장 큰 얼굴을 검출한 뒤,
    원본 크기로 bbox를 다시 매핑하여 크롭한 RGB 이미지를 반환.
    """
    h, w, _ = img_bgr.shape
    scale = 0.25
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rh, rw, _ = small_rgb.shape

    results = detector.process(small_rgb)
    if results.detections:
        # 가장 면적이 큰 얼굴 선택
        bboxes = [d.location_data.relative_bounding_box for d in results.detections]
        best = max(bboxes, key=lambda bb: bb.width * bb.height)
        x1 = max(int((best.xmin * rw) / scale), 0)
        y1 = max(int((best.ymin * rh) / scale), 0)
        x2 = min(x1 + int((best.width * rw) / scale), w)
        y2 = min(y1 + int((best.height * rh) / scale), h)
        face = img_bgr[y1:y2, x1:x2]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # 검출 실패 시 원본 전체를 RGB로 반환
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if fallback_to_full else None

# MediaPipe 얼굴 검출기 초기화
face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# 입출력 경로 설정
input_base  = r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30"
output_base = input_base + "_face_crop"
os.makedirs(output_base, exist_ok=True)

# 그룹별(폴더별) 이미지 순회 및 얼굴 크롭 저장
for group_name in os.listdir(input_base):
    src_group = os.path.join(input_base, group_name)
    if not os.path.isdir(src_group):
        continue

    dst_group = os.path.join(output_base, group_name)
    os.makedirs(dst_group, exist_ok=True)

    saved_count = 0
    for img_name in sorted(os.listdir(src_group)):
        if not img_name.lower().endswith(".png"):
            continue

        src_path = os.path.join(src_group, img_name)
        # WSL 환경이면 경로 변환
        src_path_conv = windows_path_to_wsl(src_path)

        img_bgr = robust_imread(src_path_conv)
        if img_bgr is None:
            print(f"[오류] 불러오기 실패: {src_path}")
            continue

        # 얼굴 크롭 후 BGR로 변환하여 저장
        face_rgb = crop_face(img_bgr, face_detector)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        dst_path = os.path.join(dst_group, img_name)
        cv2.imwrite(dst_path, face_bgr)
        saved_count += 1

    print(f"[완료] {group_name}: {saved_count}개 얼굴 크롭 이미지 저장됨")
