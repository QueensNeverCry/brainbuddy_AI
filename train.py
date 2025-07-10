import os
import csv
import pandas as pd
from tqdm import tqdm

def load_labels(label_csv_path):
    df = pd.read_csv(label_csv_path)
    label_dict = dict(zip(df['ClipID'].astype(str), df['binary_label']))
    return label_dict

def get_labeled_video_paths(train_txt_path, video_root, label_csv_path):
    # 라벨 불러오기
    label_dict = load_labels(label_csv_path)

    # 결과 저장 리스트
    frames_label_pairs = []

    # 파일 리스트 읽기
    with open(train_txt_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    for filename in tqdm(filenames, desc="영상 처리 중...뿅"):
        file_id = os.path.splitext(filename)[0]  # "1100062016"
        user_id = file_id[:6]  # 사용자 ID
        file_folder = file_id  # 폴더 이름
        frame_path = os.path.join(video_root, user_id, file_folder)

        if not os.path.exists(frame_path):
            print(f"❌ 프레임 없음: {frame_path}")
            continue

        # 라벨 가져오기
        label = label_dict.get(file_id)
        if label is None:
            print(f"❓ 라벨 없음: {file_id}")
            continue

        # (경로, 라벨) 저장
        frames_label_pairs.append((frame_path, label))

    return frames_label_pairs

if __name__ == "__main__":
    train_txt_path = "Train.txt"
    video_root = "extracted_frames"
    label_csv_path = "TrainLabels.csv"

    frames_label_pairs = get_labeled_video_paths(train_txt_path, video_root, label_csv_path)

    # 예시: 일부 확인
    print("예시 출력:", frames_label_pairs[:5])
