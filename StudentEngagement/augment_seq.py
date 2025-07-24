import os
import pickle
import numpy as np

def add_gaussian_noise(sequence, noise_level=0.01):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def augment_sequence(sequence, num_augment=5):
    augmented = []
    for _ in range(num_augment):
        # 1) 노이즈 추가
        noisy_seq = add_gaussian_noise(np.array(sequence))
        # 2) 시퀀스 순서 약간 섞기 (옵션)
        # 예) 앞뒤 10% 구간만 랜덤 섞기
        length = len(noisy_seq)
        idx_range = int(length * 0.1)
        idxs = list(range(length))
        np.random.shuffle(idxs[:idx_range])
        np.random.shuffle(idxs[-idx_range:])
        shuffled_seq = noisy_seq[idxs]
        augmented.append(shuffled_seq)
    return augmented

def load_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, pkl_path):
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def augment_features(input_pkl, output_pkl, augment_num_per_seq=3):
    data = load_pickle(input_pkl)
    augmented_data = {}

    for folder_name, segments in data.items():
        augmented_segments = []
        for segment in segments:
            augmented_segments.append(segment)  # 원본 시퀀스 유지
            aug_seqs = augment_sequence(segment, num_augment=augment_num_per_seq)
            augmented_segments.extend(aug_seqs)  # 증강 시퀀스 추가
        augmented_data[folder_name] = augmented_segments

    save_pickle(augmented_data, output_pkl)
    print(f"증강 완료, 저장: {output_pkl}")

if __name__ == "__main__":
    input_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features.pkl"
    output_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features_aug.pkl"
    augment_features(input_path, output_path, augment_num_per_seq=3)
