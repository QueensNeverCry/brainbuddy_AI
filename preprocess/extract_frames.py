import os
import cv2
from tqdm import tqdm
def extract_frames_from_video(video_path, output_dir, target_fps=30, max_frames=300):# 몇 프레임마다 한 장씩 저장할지 (기본값: 6)
    """
    하나의 .avi 영상에서 지정된 간격으로 프레임을 추출하여 저장
    """
    cap = cv2.VideoCapture(video_path) # video_path: 추출 대상 비디오 파일 경로
    os.makedirs(output_dir, exist_ok=True) #output_dir: 프레임을 저장할 디렉토리
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"⚠️ FPS 정보를 가져올 수 없습니다: {video_path}")
        return
    
    frame_interval = max(int(fps / target_fps), 1)
    #print(f"📹 {os.path.basename(video_path)} - FPS: {fps:.2f}, Interval: {frame_interval}")
    
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved>=max_frames: #300장 이상 추출되면 넘어감
            break

        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"{saved:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1
        count += 1

    cap.release()
    return saved

def extract_all_from_train_txt(train_txt_path, video_root, output_root):
    with open(train_txt_path, 'r') as f:
        filenames = [line.strip() for line in f.readlines()]

    total_videos = 0        # 전체 존재하는 영상 수
    processed_videos = 0    # 프레임 추출된 영상 수

    for filename in tqdm(filenames, desc="영상처리중...뿅"):
        file_id = os.path.splitext(filename)[0]
        user_id = file_id[:6]

        video_path = os.path.join(video_root, user_id, file_id, filename)
        output_dir = os.path.join(output_root, user_id, file_id)

        if not os.path.exists(video_path):
            print(f"❌ 영상 없음: {video_path}")
            continue

        total_videos += 1
        saved_frames = extract_frames_from_video(video_path, output_dir, target_fps=30)
        if saved_frames > 0:
            processed_videos += 1

    print(f"총 영상 파일 수: {total_videos}")
    print(f"프레임 추출 완료된 영상 수: {processed_videos}")


if __name__ == "__main__":
    train_txt_path = "../DataSet/Train.txt"
    val_txt_path ="../DataSet/Validation.txt"
    
    train_video_root="../DataSet/Train"
    val_video_root = "../DataSet/Validation"
    
    train_output_root="frames/train"
    val_output_root = "frames/validation"
    
    extract_all_from_train_txt(train_txt_path, train_video_root, train_output_root)
    extract_all_from_train_txt(val_txt_path, val_video_root, val_output_root)

