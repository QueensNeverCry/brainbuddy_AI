import pickle
from collections import Counter


#label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/test/test_features.pkl"
label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/test/test_features_aug.pkl"
#label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features.pkl"
#label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features_aug.pkl"

#label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/label/test_labels.pkl"
#label_path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/label/train_labels.pkl"



with open(label_path, "rb") as f:
    data = pickle.load(f)

# data 구조 예시
# {
#   'Engaged폴더명1': [ [벡터1, 벡터2, ...], [벡터1, 벡터2, ...], ... ], 
#   'Not engaged폴더명1': [ [...], [...], ... ],
#   ...
# }
# key는 세부 폴더명, value는 리스트(세그먼트)들의 리스트(각 세그먼트는 이미지 벡터 리스트)

engaged_count = 0
not_engaged_count = 0

for folder_name, segments in data.items():
    if folder_name.lower().startswith("engaged"):
        engaged_count += len(segments)
    elif folder_name.lower().startswith("not engaged") or folder_name.lower().startswith("not_engaged"):
        not_engaged_count += len(segments)

print(f"Engaged: {engaged_count}")
print(f"Not engaged: {not_engaged_count}")
print(f"Total: {engaged_count + not_engaged_count}")
