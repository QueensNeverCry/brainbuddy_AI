import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# 폴더명에서 라벨 추출: 끝 2자리
def extract_label_from_folder(folder_name):
    label_code = folder_name[-2:]
    return 1 if label_code == '01' else 0  # 01: 집중(1), 03: 비집중(0)

# 프레임 간 평균 픽셀 차이 계산
def compute_motion_scores(segment_path, resize=(224, 224)):
    files = sorted(os.listdir(segment_path))
    files = [f for f in files if f.endswith('.jpg')]
    scores = []

    for i in range(len(files) - 1):
        f1 = cv2.imread(os.path.join(segment_path, files[i]), cv2.IMREAD_GRAYSCALE)
        f2 = cv2.imread(os.path.join(segment_path, files[i + 1]), cv2.IMREAD_GRAYSCALE)

        # 리사이즈로 강제 통일
        f1 = cv2.resize(f1, resize)
        f2 = cv2.resize(f2, resize)

        diff = np.abs(f1.astype(np.float32) - f2.astype(np.float32))
        scores.append(np.mean(diff))

    return scores


# 기본 경로
base_dir = r"C:/AIhub_frames/train"

# 결과 저장 리스트
avg_motion_focus = []
avg_motion_nonfocus = []

folder_list = os.listdir(base_dir)

for folder_name in tqdm(folder_list, desc="Processing sequences"):
    folder_path = os.path.join(base_dir, folder_name)
    segment_path = os.path.join(folder_path, "segment_0")
    
    if not os.path.exists(segment_path):
        continue

    label = extract_label_from_folder(folder_name)
    motion_scores = compute_motion_scores(segment_path)

    if not motion_scores:
        continue

    avg_motion = np.mean(motion_scores)

    if label == 1:
        avg_motion_focus.append(avg_motion)
    else:
        avg_motion_nonfocus.append(avg_motion)

# 시각화 (시퀀스별 평균 움직임 플롯)
plt.figure(figsize=(10, 5))
plt.scatter(range(len(avg_motion_focus)), avg_motion_focus, color='blue', label='Focus (1)', alpha=0.7)
plt.scatter(range(len(avg_motion_focus), len(avg_motion_focus) + len(avg_motion_nonfocus)), avg_motion_nonfocus, color='orange', label='Non-Focus (0)', alpha=0.7)

plt.title("Average Motion per Sequence")
plt.xlabel("Sequence Index")
plt.ylabel("Average Frame-to-Frame Pixel Difference")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
