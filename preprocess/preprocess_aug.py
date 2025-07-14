import pickle
import os
from preprocess_and_save_features import preprocess_dataset  # 기존 전처리 함수 불러오기

def load_extended_dataset(original_pickle_path, original_root="train_frames", aug_root="train_aug_frames"):
    with open(original_pickle_path, "rb") as f:
        dataset_links = pickle.load(f)

    augmented_links = []
    for path, label in dataset_links:
        if label == 0 and original_root in path:
            aug_path = path.replace(original_root, aug_root) + "_aug"
            if os.path.exists(aug_path):
                augmented_links.append((aug_path, label))
            else:
                print(f"❗ 증강 폴더 없음: {aug_path}")
    
    print(f"✅ 증강된 라벨 0 데이터 수: {len(augmented_links)}")
    return dataset_links + augmented_links

def count_label_zero(dataset_links):
    return sum(1 for _, label in dataset_links if label == 0)

if __name__ == "__main__":
    # 🟡 원본 + 증강 학습 데이터 병합
    extended_train_link = load_extended_dataset(
        original_pickle_path="train_link.pkl",
        original_root="train_frames",
        aug_root="train_aug_frames"
    )

    # 🟡 Validation은 그대로 사용
    with open("val_link.pkl", "rb") as f:
        val_link = pickle.load(f)

    # 🟢 CNN Feature 전처리 저장
    preprocess_dataset(extended_train_link, save_dir="preprocessed_features/train_data", T=10)
    preprocess_dataset(val_link, save_dir="preprocessed_features/val_data", T=10)

    # ✅ 출력
    print(f"✅ 최종 학습 데이터 수: {len(extended_train_link)}개")
    print(f"✅ 최종 검증 데이터 수: {len(val_link)}개")
    print(f"✅ 최종 학습 데이터 중 라벨 0 개수: {count_label_zero(extended_train_link)}개")
