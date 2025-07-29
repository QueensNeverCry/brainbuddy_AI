import os
import shutil
from collections import defaultdict

target_displays=["Laptop","Monitor"]
# 원본 png 파일들이 있는 경로
base_root = r"C:\Users\user\Downloads\126.디스플레이 중심 안구 움직임 영상 데이터\01-1.정식개방데이터\Training\01.원천데이터\TS.zip\001\T1"

# 결과 폴더
output_base = os.path.join(base_root, "분류된_이미지")
os.makedirs(output_base, exist_ok=True)

# 공통 이름 기준으로 파일 분류
grouped_files = defaultdict(list)

for display in target_displays :
    rgb_path = os.path.join(base_root, display, "rgb")
    if not os.path.isdir(rgb_path):
        print(f"[경고] 경로 없음: {rgb_path}")
        continue
    for filename in os.listdir(rgb_path):
        if not filename.endswith(".png"):
            continue
        base = filename.rsplit("_",1)[0]
        grouped_files[base].append((filename, rgb_path))
# grouped_files = {
#     "S01_S_D_E_T": [
#         ("S01_S_D_E_T_00001.png", "...\\Laptop\\rgb"),
#         ("S01_S_D_E_T_00002.png", "...\\Laptop\\rgb"),
#         ("S01_S_D_E_T_00003.png", "...\\Monitor\\rgb"),  # 예시
#         ...
#     ],
#     "S02_S_C_D_T": [
#         ("S02_S_C_D_T_00001.png", "...\\Laptop\\rgb"),
#         ...
#     ]
# }

# 파일 이동 및 폴더 생성
for group_name, file_list in grouped_files.items(): # key, value 쌍을 하나씩 가져옴 - key :  "S01_S_D_E_T", value(file list) : [(파일명, 경로), (파일명, 경로), ...]
    group_folder = os.path.join(output_base, group_name)
    os.makedirs(group_folder, exist_ok=True)
    for fname, src_dir in file_list:
        src_file = os.path.join(src_dir, fname)
        dst_file = os.path.join(group_folder, fname)
        shutil.move(src_file, dst_file)

# 결과 출력
print("\n=== 그룹별 PNG 파일 개수 ===")
for group_name, files in grouped_files.items():
    print(f"{group_name}: {len(files)}개")
