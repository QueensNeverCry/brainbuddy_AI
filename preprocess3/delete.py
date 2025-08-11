import os

# 대상 최상위 경로 설정
BASE_DIR = r"C:/AIhub_frames/test"
TARGET_FILENAME = "fusion_features.pkl"

def delete_all_fusion_features(base_dir):
    deleted_count = 0

    for subject_folder in os.listdir(base_dir):
        subject_path = os.path.join(base_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for segment_folder in os.listdir(subject_path):
            segment_path = os.path.join(subject_path, segment_folder)
            if not os.path.isdir(segment_path):
                continue

            target_path = os.path.join(segment_path, TARGET_FILENAME)
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                    print(f"🗑️ 삭제됨: {target_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ 삭제 실패: {target_path} ({e})")

    print(f"\n✅ 총 {deleted_count}개 fusion_features.pkl 삭제 완료.")

if __name__ == "__main__":
    delete_all_fusion_features(BASE_DIR)

