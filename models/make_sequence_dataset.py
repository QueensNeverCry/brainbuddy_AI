import os
import numpy as np
import pickle
from tqdm import tqdm

def create_sequence_dataset(feature_root, label_pkl_path, sequence_length=60, stride=30, save_path="sequence_dataset.pkl"):
    with open(label_pkl_path, "rb") as f:
        dataset_info = pickle.load(f)

    sequences = []
    sequence_labels = []

    for clip_dir, label in tqdm(dataset_info, desc="📦 시퀀스 생성 중"):
        if not os.path.exists(clip_dir):
            continue

        feature_files = sorted([
            os.path.join(clip_dir, f) for f in os.listdir(clip_dir)
            if f.endswith(".npy")
        ])

        if len(feature_files) < sequence_length:
            continue

        feature_array = [np.load(f) for f in feature_files]
        feature_array = np.stack(feature_array)

        for start in range(0, len(feature_array) - sequence_length + 1, stride):
            window = feature_array[start:start + sequence_length]
            sequences.append(window)
            sequence_labels.append(label)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "sequences": sequences,
            "labels": sequence_labels
        }, f)

    print(f"✅ 저장 완료: {save_path}, 총 시퀀스 수: {len(sequences)}")

if __name__ == "__main__":
    feature_root = "C:/KSEB/brainbuddy_AI/models/features"
    
    create_sequence_dataset(
        feature_root=os.path.join(feature_root, "train"),
        label_pkl_path="train_link.pkl",
        sequence_length=60,
        stride=30,
        save_path="C:/KSEB/brainbuddy_AI/models/sequences/train_sequence.pkl"
    )

    create_sequence_dataset(
        feature_root=os.path.join(feature_root, "valid"),
        label_pkl_path="val_link.pkl",
        sequence_length=60,
        stride=30,
        save_path="C:/KSEB/brainbuddy_AI/models/sequences/val_sequence.pkl"
    )
