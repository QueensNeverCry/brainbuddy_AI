# 라벨이 0인 데이터만 증강
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import random
import pickle

# 프레임 단위 증강 함수들
def add_gaussian_noise(image, mean=0, sigma=10):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
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
    do_noise = random.random() < 0.4
    noise_sigma = random.randint(5, 15) if do_noise else 0

    do_brightness_contrast = random.random() < 0.5
    brightness = random.randint(-40, 40) if do_brightness_contrast else 0
    contrast = random.randint(-40, 40) if do_brightness_contrast else 0

    do_flip = random.random() < 0.6

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
    if len(frame_paths) < 10:
        print(f"[경고] 프레임이 10개 미만: {input_dir} ({len(frame_paths)}개)")
        return

    # 앞에서 10개만 사용
    frame_paths = frame_paths[:10]

    frames = [cv2.imread(fp) for fp in frame_paths]
    augmented_frames = augment_sequence(frames)

    for i, aug_frame in enumerate(augmented_frames):
        save_path = os.path.join(output_dir, f"{i:04d}.jpg")
        cv2.imwrite(save_path, aug_frame)

# pickle에서 (폴더경로, 라벨) 불러와 라벨 0인 데이터만 증강
def augment_from_label_pickle(pickle_path, output_root,input_root):
    with open(pickle_path, "rb") as f:
        dataset_links = pickle.load(f)  # [(폴더경로, 라벨), ...]

    zero_label_count = 0
    for seq_dir, label in tqdm(dataset_links, desc="라벨 0 증강 중"):
        if label == 0:
            zero_label_count += 1
            relative_path = os.path.relpath(seq_dir, start=input_root)
            #print(relative_path)
            output_dir = os.path.join(output_root, relative_path + "_aug")
            #print("out",output_dir)
            augment_sequence_folder(seq_dir, output_dir)

    print(f"라벨 0인 시퀀스 총 {zero_label_count}개 증강 완료")

if __name__ == "__main__":
    train_pickle = "../preprocessed/train_link.pkl"
    val_pickle = "../preprocessed/val_link.pkl"

    train_output_root = "../frames/train_aug_frames"
    val_output_root = "../frames/val_aug_frames"
    # input_root : 기준 경로
    train_input_root = "../frames/train_frames"
    val_input_root ="../frames/valid_frames"
    
    print("=== Train 데이터 라벨 0 증강 ===")
    augment_from_label_pickle(train_pickle, train_output_root,train_input_root)

    print("=== Validation 데이터 라벨 0 증강 ===")
    augment_from_label_pickle(val_pickle, val_output_root,val_input_root)
