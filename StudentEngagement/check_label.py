import pickle

label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/label/train_labels_aug.pkl"

with open(label_path, "rb") as f:
    labels = pickle.load(f)

print(f"총 샘플 수: {len(labels)}")

# 폴더명별 샘플 개수 집계
from collections import Counter
folder_counter = Counter()
label_counter = Counter()

for folder_name, label in labels:
    folder_counter[folder_name] += 1
    label_counter[label] += 1

print("폴더별 샘플 개수:")
for folder, count in folder_counter.items():
    print(f"  {folder}: {count}")

print("라벨별 샘플 개수:")
for label, count in label_counter.items():
    print(f"  라벨 {label}: {count}")
