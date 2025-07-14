import os
import cv2
from tqdm import tqdm
def extract_frames_from_video(video_path, output_dir, frame_interval):# 몇 프레임마다 한 장씩 저장할지 (기본값: 6)
    """
    하나의 .avi 영상에서 지정된 간격으로 프레임을 추출하여 저장
    """
    cap = cv2.VideoCapture(video_path) # video_path: 추출 대상 비디오 파일 경로
    os.makedirs(output_dir, exist_ok=True) #output_dir: 프레임을 저장할 디렉토리

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
        count += 1

    cap.release()
    #print(f"✅ {os.path.basename(video_path)} → {saved}장 저장됨")


def extract_all_from_train_txt(train_txt_path, video_root, output_root, frame_interval,total):
    with open(train_txt_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    for filename in tqdm(filenames,desc="영상처리중...뿅"):
        file_id = os.path.splitext(filename)[0]  # "1100062016"
        user_id = file_id[:6]  # 앞 6자리: 사용자 ID "110006"

        # 영상 경로: DataSet/Train/001/001003/001003.avi
        video_path = os.path.join(video_root, user_id, file_id, filename)

        # 저장 경로: extracted_frames/001/001003/
        output_dir = os.path.join(output_root, user_id, file_id)

        if not os.path.exists(video_path):
            print(f"❌ 영상 없음: {video_path}")
            continue
        else:
            total+=1
        extract_frames_from_video(video_path, output_dir, frame_interval=frame_interval)
    print(total)




if __name__ == "__main__":
    #train으로 바꾸면 train dataset 전처리 할 수 있음 
    train_txt_path = "../DataSet/Validation.txt"
    video_root = "../DataSet/Validation"
    output_root = "validation"
    frame_interval = 30  # 6프레임마다 1장
    total=0
    extract_all_from_train_txt(train_txt_path, video_root, output_root, frame_interval,total)
