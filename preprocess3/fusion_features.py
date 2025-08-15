import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# 설정
BASE_DIR = r"C:/AIhub_frames/train"
FUSION_FEATURE_DIM = 5
IMAGE_SIZE = (640, 360)

# Mediapipe 설정 (global로 선언 불가, 프로세스 내에서 초기화해야 함)
def init_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# 각 segment 처리 함수
def process_segment(segment_path):
    try:
        fusion_feat_path = os.path.join(segment_path, "fusion_features.pkl")
        if os.path.exists(fusion_feat_path):
            return None  # 이미 존재하면 스킵

        image_files = sorted([
            f for f in os.listdir(segment_path)
            if f.lower().endswith('.jpg') and f[:4].isdigit()
        ])

        if len(image_files) < 30:
            return f"⚠️ {segment_path} - 프레임 부족 ({len(image_files)})"

        image_files = image_files[:30]
        face_mesh = init_face_mesh()

        yaw_list, pitch_list, head_movement = [], [], []
        eye_closed_frames = 0
        yawn_detected = 0
        prev_nose = None

        for fname in image_files:
            img_path = os.path.join(segment_path, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark

            # 머리 이동
            nose = landmarks[1]
            if prev_nose is not None:
                dx = nose.x - prev_nose.x
                dy = nose.y - prev_nose.y
                head_movement.append(np.sqrt(dx**2 + dy**2))
            prev_nose = nose

            # 하품
            lip_dist = abs(landmarks[13].y - landmarks[14].y)
            if lip_dist > 0.05:
                yawn_detected = 1

            # 눈 감김
            eye_openness = abs(landmarks[159].y - landmarks[145].y)
            if eye_openness < 0.015:
                eye_closed_frames += 1

            # 얼굴 각도
            yaw = landmarks[263].x - landmarks[33].x
            pitch = landmarks[152].y - landmarks[10].y
            yaw_list.append(yaw)
            pitch_list.append(pitch)

        # 통계량 계산
        yaw_diff = np.std(np.diff(yaw_list)) if len(yaw_list) > 1 else 0
        pitch_diff = np.std(np.diff(pitch_list)) if len(pitch_list) > 1 else 0
        eye_closed_ratio = eye_closed_frames / 30
        head_speed = np.mean(head_movement) if head_movement else 0

        features = [yaw_diff, pitch_diff, float(yawn_detected), eye_closed_ratio, head_speed]
        if len(features) != FUSION_FEATURE_DIM:
            return f"❌ {segment_path} - feature 길이 오류"

        with open(fusion_feat_path, 'wb') as f:
            pickle.dump(features, f)

        return None  # 성공

    except Exception as e:
        return f"❌ {segment_path} - 처리 중 오류: {e}"


# 전체 segment 경로 수집
def get_all_segment_paths(base_dir):
    all_segments = []
    for subject_folder in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        for segment_folder in os.listdir(subject_path):
            segment_path = os.path.join(subject_path, segment_folder)
            if os.path.isdir(segment_path):
                all_segments.append(segment_path)
    return all_segments


# 메인 실행
def main():
    segment_paths = get_all_segment_paths(BASE_DIR)
    print(f"📂 총 segment 수: {len(segment_paths)}")
    errors = []

    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_segment, segment_paths), total=len(segment_paths)):
            if result is not None:
                errors.append(result)

    print("\n✅ 전처리 완료!")
    if errors:
        print(f"\n⚠️ 오류/스킵된 항목 {len(errors)}개:")
        for e in errors:
            print(e)

if __name__ == "__main__":
    main()
