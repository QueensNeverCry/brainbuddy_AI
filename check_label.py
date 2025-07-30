import os
import pickle

# 파일 불러오기
with open('C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"📦 원래 데이터 개수 (pkl 내): {len(data)}")
missing = []
for path, _ in data:
    if not os.path.isdir(path):
        missing.append(path)

print(f"❌ 존재하지 않는 폴더 수: {len(missing)}")
