import cv2
import os
import shutil
from tqdm import tqdm
from models.face_crop import crop_face
import mediapipe as mp

def extract_frames(video_path, output_dir, face_detector, target_fps=10, max_frames=100):
    cap = cv2.VideoCapture(video_path)

    os.makedirs(output_dir, exist_ok=True) #output_dir: 프레임을 저장할 디렉토리
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"⚠️ FPS 정보를 가져올 수 없습니다: {video_path}")
        return
    
    frame_interval = max(int(fps / target_fps), 1)
    
    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or saved>=max_frames: #100장 이상 추출되면 넘어감
            break

        if count % frame_interval == 0:
            cropped = crop_face(frame, face_detector)
            if cropped is not None:
                frame_path = os.path.join(output_dir, f"{saved:04d}.jpg")
                cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                success = cv2.imwrite(frame_path, cropped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if success :
                    saved += 1
        count += 1

    cap.release()
    return saved

def extract_all_from_train_txt(train_txt_path, video_root, output_root, face_detector):
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
        saved_frames = extract_frames(video_path, output_dir, face_detector, target_fps=10, max_frames=100)
        if saved_frames > 0:
            processed_videos += 1

    print(f"총 영상 파일 수: {total_videos}")
    print(f"프레임 추출 완료된 영상 수: {processed_videos}")


if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    #train_txt_path = "C:/Users/user/Desktop/KSEB/Dataset_/DAiSEE/DataSet/Train.txt"
    val_txt_path = "C:/Users/user/Desktop/KSEB/Dataset_/DAiSEE/DataSet/Validation.txt"
    
    #train_video_root="C:/Users/user/Desktop/KSEB/Dataset_/DAiSEE/DataSet/Train"
    val_video_root = "C:/Users/user/Desktop/KSEB/Dataset_/DAiSEE/DataSet/Validation"
    
    #train_output_root="C:/DAiSEE/train"
    val_output_root = "C:/DAiSEE/valid"

    #extract_all_from_train_txt(train_txt_path, train_video_root, train_output_root, face_detector)
    extract_all_from_train_txt(val_txt_path, val_video_root, val_output_root, face_detector)

    face_detector.close()