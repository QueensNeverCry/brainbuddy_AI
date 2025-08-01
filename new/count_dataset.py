# 데이터셋 몇개나 만들었는지 체크하기
import os
from glob import glob

train_root = r"C:/eye_dataset/valid"
valid_count = 0

for folder_name in os.listdir(train_root):
    folder_path = os.path.join(train_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    json_files = glob(os.path.join(folder_path, "*.json"))
    if len(json_files) == 30:
        valid_count += 1

print(f"📦 현재 라벨링 완료된 데이터셋 수: {valid_count}")
