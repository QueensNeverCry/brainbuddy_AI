import pickle
import os

def create_labels_from_augmented_pickle(pickle_path, save_path):
    with open(pickle_path, "rb") as f:
        features = pickle.load(f)

    # 폴더명별 라벨 매핑 (필요하면 변경)
    label_map = {
        "Engaged": 1,
        "Not engaged": 0
    }

    labels = []

    for folder_name, segments in features.items():
        label = label_map.get(folder_name, -1)
        if label == -1:
            print(f"⚠️ 경고: 폴더명 '{folder_name}' 에 매핑된 라벨 없음")
            continue

        for segment in segments:
            # segment 하나가 하나의 샘플 단위
            labels.append((folder_name, label))

    with open(save_path, "wb") as f:
        pickle.dump(labels, f)

    print(f"라벨링 완료, 저장 위치: {save_path}")
    print(f"총 샘플 수: {len(labels)}")

if __name__ == "__main__":
    pickle_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features_aug.pkl"
    save_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/label/train_labels_aug.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    create_labels_from_augmented_pickle(pickle_path, save_path)
