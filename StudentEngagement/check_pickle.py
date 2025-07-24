import pickle

path = r"C:/Users/user/Desktop/brainbuddy_AI/StudentEngagement/features/train/train_features_aug.pkl"

with open(path, "rb") as f:
    features = pickle.load(f)

print(f"폴더(키) 개수: {len(features)}")

for folder, segments in features.items():
    print(f"\n폴더명: {folder}")
    print(f"  segment 개수: {len(segments)}")
    for i, segment in enumerate(segments):
        print(f"    segment {i+1} 이미지 벡터 개수: {len(segment)}")
