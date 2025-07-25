import os
import pickle
import numpy as np

def add_gaussian_noise(sequence, noise_level):
    noise = np.random.normal(0, noise_level, sequence.shape)
    return sequence + noise

def augment_sequence(sequence, noise_levels, num_augment_per_level=3):
    augmented = []
    for noise_level in noise_levels:
        for _ in range(num_augment_per_level):
            noisy_seq = add_gaussian_noise(np.array(sequence), noise_level)
            length = len(noisy_seq)
            idx_range = int(length * 0.1)
            idxs = list(range(length))
            np.random.shuffle(idxs[:idx_range])
            np.random.shuffle(idxs[-idx_range:])
            shuffled_seq = noisy_seq[idxs]
            shuffled_seq = shuffled_seq.astype(np.float32)
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

def augment_features(input_pkl, output_pkl):
    noise_levels = [0.001, 0.003, 0.005, 0.008, 0.010, 0.013, 0.015, 0.018, 0.02]
    num_augment_per_level = 3

    data = load_pickle(input_pkl)
    augmented_data = {}

    for folder_name, segments in data.items():
        augmented_segments = []
        for segment in segments:
            augmented_segments.append(segment)  # 원본 유지
            aug_seqs = augment_sequence(segment, noise_levels, num_augment_per_level)
            augmented_segments.extend(aug_seqs)
        augmented_data[folder_name] = augmented_segments

    save_pickle(augmented_data, output_pkl)
    print(f"증강 완료, 저장: {output_pkl}")

if __name__ == "__main__":
    input_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/test/test_features.pkl"
    output_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/test/test_features_aug.pkl"
    augment_features(input_path, output_path)
