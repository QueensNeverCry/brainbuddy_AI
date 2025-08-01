#json 데이터셋을 같은 라벨링 단위로 30개씩 묶어 폴더에 저장
import os
import shutil
from glob import glob

# 경로 설정
label_root = r"C:/Users/user/Downloads/126.디스플레이 중심 안구 움직임 영상 데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL"
output_root = r"C:/eye_dataset/valid"
devices = ["Monitor", "Laptop"]
json_subdir = "json_rgb"
max_count = 30
min_count = 11  # 10개 이하면 제거

# 순회
for seq in range(1, 149):
    seq_str = f"{seq:03d}"
    for device in devices:
        json_dir = os.path.join(label_root, seq_str, "T1", device, json_subdir)
        if not os.path.exists(json_dir):
            continue

        json_files = glob(os.path.join(json_dir, "*.json"))
        if not json_files:
            continue

        # prefix 기준 그룹핑
        label_groups = {}
        for file in json_files:
            filename = os.path.basename(file)
            parts = filename.split("_")
            
            if len(parts) < 4:
                continue  # 예외 처리

            posture = parts[-3]  # 뒤에서 세 번째
            if posture not in ["C", "D", "H", "T", "U"]:
                continue  # 조건에 맞는 자세만 포함

            prefix = "_".join(parts[:-1])  # 프레임 번호 제외한 prefix
            label_groups.setdefault(prefix, []).append(file)

        for prefix, files in label_groups.items():
            files.sort(key=lambda x: int(os.path.splitext(x)[0].split("_")[-1]))
            file_count = len(files)

            # 10개 이하면 건너뜀
            if file_count <= 10:
                print(f"⚠️ 건너뜀 (10개 이하): {prefix}")
                continue

            # 복사 리스트 구성
            extended_files = []

            # 앞뒤에서 2개씩 복사
            if file_count < max_count:
                front = files[:2] if file_count >= 2 else files[:1]
                back = files[-2:] if file_count >= 2 else files[-1:]

                extended_files.extend(front)
                extended_files.extend(files)
                extended_files.extend(back)

                # 부족하면 마지막 파일로 채우기
                while len(extended_files) < max_count:
                    extended_files.append(files[-1])
            else:
                extended_files = files[:max_count]  # 30개 이상이면 자르기

            # 폴더 생성 및 저장
            target_folder = os.path.join(output_root, prefix)
            os.makedirs(target_folder, exist_ok=True)

            for i, f in enumerate(extended_files):
                target_path = os.path.join(target_folder, f"{i:03d}.json")
                shutil.copy2(f, target_path)

            print(f"✅ 저장 완료: {prefix} → {len(extended_files)}개 → {target_folder}")
