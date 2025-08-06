import os
import cv2
import mediapipe as mp
import numpy as np
import platform
import re

def windows_path_to_wsl(path: str) -> str:
    if platform.system() == "Linux" and re.match(r"^[A-Za-z]:\\", path):
        drive, rest = path.split(":", 1)
        rest = rest.replace("\\", "/")
        return f"/mnt/{drive.lower()}{rest}"
    return path

def robust_imread(path: str):
    p = os.path.normpath(path)
    try:
        with open(p, "rb") as f:
            data = f.read()
        nparr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    img = cv2.imread(p)
    if img is not None:
        return img
    if os.name == "nt":
        unc = r"\\\\?\\" + p
        img = cv2.imread(unc)
        if img is not None:
            return img
    return None

def crop_face(img_bgr, detector, fallback_to_full=True):
    h, w, _ = img_bgr.shape
    scale = 0.25
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rh, rw, _ = small_rgb.shape

    results = detector.process(small_rgb)
    if results.detections:
        bboxes = [d.location_data.relative_bounding_box for d in results.detections]
        best = max(bboxes, key=lambda bb: bb.width * bb.height)
        x1 = max(int((best.xmin * rw) / scale), 0)
        y1 = max(int((best.ymin * rh) / scale), 0)
        x2 = min(x1 + int((best.width * rw) / scale), w)
        y2 = min(y1 + int((best.height * rh) / scale), h)
        face = img_bgr[y1:y2, x1:x2]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if fallback_to_full else None

def process_folder(input_base: str):
    # MediaPipe 얼굴 검출기
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    output_base = input_base + "_face_crop"
    os.makedirs(output_base, exist_ok=True)

    # 라벨(클래스)별 서브폴더 순회
    for label in sorted(os.listdir(input_base)):
        src_label_dir = os.path.join(input_base, label)
        if not os.path.isdir(src_label_dir):
            continue

        dst_label_dir = os.path.join(output_base, label)
        os.makedirs(dst_label_dir, exist_ok=True)

        saved_count = 0
        for img_name in sorted(os.listdir(src_label_dir)):
            if not img_name.lower().endswith(".png"):
                continue

            src_path = os.path.join(src_label_dir, img_name)
            img_bgr = robust_imread(windows_path_to_wsl(src_path))
            if img_bgr is None:
                print(f"[오류] 읽기 실패: {src_path}")
                continue

            face_rgb = crop_face(img_bgr, face_detector)
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_label_dir, img_name), face_bgr)
            saved_count += 1

        print(f"[완료] {os.path.basename(input_base)}/{label}: {saved_count}개 저장됨")

def main():
    # 처리할 세 개의 폴더 경로 리스트
    input_dirs = [
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134",
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135",
        r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136",
    ]
    for d in input_dirs:
        process_folder(d)

if __name__ == "__main__":
    main()
