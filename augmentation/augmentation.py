# 전체 데이터를 증강하는 코드
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random

# 프레임 단위 증강 함수들
def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    return noisy

def adjust_brightness_contrast(image, brightness=30, contrast=30):
    beta = brightness
    alpha = 1 + (contrast / 100.0)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def horizontal_flip(image):
    return cv2.flip(image, 1)

# 시퀀스 단위 증강 함수
def augment_sequence(frames):
    # 시퀀스 단위 증강 파라미터 랜덤 결정
    do_noise = random.random() < 0.5
    noise_sigma = random.randint(5, 15) if do_noise else 0

    do_brightness_contrast = random.random() < 0.5
    brightness = random.randint(-40, 40) if do_brightness_contrast else 0
    contrast = random.randint(-40, 40) if do_brightness_contrast else 0

    do_flip = random.random() < 0.5

    augmented_frames = []
    for frame in frames:
        img = frame.copy()
        if do_noise:
            img = add_gaussian_noise(img, sigma=noise_sigma)
        if do_brightness_contrast:
            img = adjust_brightness_contrast(img, brightness, contrast)
        if do_flip:
            img = horizontal_flip(img)
        augmented_frames.append(img)
    return augmented_frames

# 하나의 시퀀스 폴더에서 프레임 읽고 증강 후 저장
def augment_sequence_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    frame_paths = sorted(glob(os.path.join(input_dir, '*.jpg')))
    assert len(frame_paths) == 10, f"프레임 수가 10개가 아님: {input_dir}"

    # 10개 프레임 읽기
    frames = [cv2.imread(fp) for fp in frame_paths]

    # 시퀀스 단위 증강 적용
    augmented_frames = augment_sequence(frames)

    # 저장
    for i, aug_frame in enumerate(augmented_frames):
        save_path = os.path.join(output_dir, f"{i:04d}.jpg")
        cv2.imwrite(save_path, aug_frame)

# 루트 폴더 내 모든 시퀀스 폴더에 증강 적용
def augment_all_sequences(input_root, output_root):
    sequence_dirs = []
    for root, dirs, files in os.walk(input_root):
        jpg_files = [f for f in files if f.endswith('.jpg')]
        if len(jpg_files) == 10:
            sequence_dirs.append(root)

    print(f"총 {len(sequence_dirs)}개 시퀀스 증강 시작...")

    for seq_dir in tqdm(sequence_dirs):
        relative_path = os.path.relpath(seq_dir, input_root)
        output_dir = os.path.join(output_root, relative_path + "_aug")
        augment_sequence_folder(seq_dir, output_dir)

if __name__ == "__main__":
    train_input = "../frames/train_frames"        # 원본 10프레임 폴더들 루트
    train_output = "../frames/train_aug_frames"   # 증강된 시퀀스 저장할 루트
    val_input = "../frames/val_frames"        # 원본 10프레임 폴더들 루트
    val_output = "../frames/val_aug_frames"  

    augment_all_sequences(train_input, train_output)
    augment_all_sequences(val_input, val_output)

