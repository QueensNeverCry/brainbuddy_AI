import os
import numpy as np
import imageio.v3 as iio
from collections import defaultdict

# ✅ 정규화 함수 (부족하면 마지막 프레임 복제해서 뒤에 붙임)
def normalize_frame_sequence_np(frames: np.ndarray, target_len=30) -> np.ndarray:
    current_len = frames.shape[0]
    if current_len >= target_len:
        return frames[:target_len]
    else:
        pad_total = target_len - current_len
        back_pad = np.repeat(frames[-1:], pad_total, axis=0)
        return np.concatenate([frames, back_pad], axis=0)

# ✅ 대상 디스플레이
target_displays = ["Laptop", "Monitor"]

# ✅ 데이터 경로
base_root = r"C:\Users\user\Downloads\126.디스플레이 중심 안구 움직임 영상 데이터\01-1.정식개방데이터\Training\01.원천데이터\TS\001\T1"
output_base = os.path.join(base_root, "분류된_이미지_정규화30")
os.makedirs(output_base, exist_ok=True)

# ✅ 파일 그룹화
grouped_files = defaultdict(list)
for display in target_displays:
    rgb_path = os.path.join(base_root, display, "rgb")
    if not os.path.isdir(rgb_path):
        print(f"[경고] 경로 없음: {rgb_path}")
        continue
    for filename in os.listdir(rgb_path):
        if not filename.endswith(".png"):
            continue
        base = filename.rsplit("_", 1)[0]  # 예: "S01_S_D_E_T_00001.png" → "S01_S_D_E_T"
        grouped_files[base].append((filename, rgb_path))

# ✅ 그룹별 처리
print("\n=== 그룹별 30프레임 정규화 및 저장 결과 ===")
for group_name, file_list in grouped_files.items():
    file_list.sort(key=lambda x: x[0])  # 파일명 정렬

    # 이미지 로딩
    frames = []
    for fname, src_dir in file_list:
        src_file = os.path.join(src_dir, fname)
        img = iio.imread(src_file)
        frames.append(img)

    frames_np = np.stack(frames, axis=0)  # shape: (N, H, W, C)

    # 30프레임 정규화
    norm_frames = normalize_frame_sequence_np(frames_np, target_len=30)

    # 저장용 폴더 생성
    group_folder = os.path.join(output_base, group_name)
    os.makedirs(group_folder, exist_ok=True)

    # 저장
    for idx, img in enumerate(norm_frames):
        save_name = f"{group_name}_{idx+1:05d}.png"
        save_path = os.path.join(group_folder, save_name)
        iio.imwrite(save_path, img)

    print(f"[완료] {group_name}: 총 {len(norm_frames)}개 저장됨 (원본: {len(frames_np)})")
