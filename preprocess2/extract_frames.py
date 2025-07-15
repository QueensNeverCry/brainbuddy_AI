import os
import cv2
import torch
from tqdm import tqdm
import mediapipe as mp

from models.face_detector import extract_face
from models.feature_extractor import extract_cnn_features

def calculate_frame_interval(video_path,target_fps):
    """
    원본 영상의 FPS를 기준으로 초당 target_fps만큼의 프레임을 저장하기 위한 frame_interval 값을 계산해주는 함수
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps == 0:
        raise ValueError("⚠️ 영상에서 FPS 정보를 읽을 수 없습니다.")

    interval = max(1, round(fps / target_fps))
    print(f"영상 FPS: {fps:.2f} → 초당 {target_fps}장 저장하려면 frame_interval = {interval}")
    return interval

def extract_frames_from_video(video_path, output_dir, frame_interval):# 몇 프레임마다 한 장씩 저장할지 (기본값: 6)
    """
    하나의 영상에서 지정된 간격으로 프레임을 10개씩 추출하여 폴더별로 저장하는 함수
    """
    cap = cv2.VideoCapture(video_path) # video_path: 추출 대상 비디오 파일 경로
    os.makedirs(output_dir, exist_ok=True) #output_dir: 프레임을 저장할 디렉토리

    count = 0          # 전체 프레임 카운터
    saved = 0          # 저장한 총 프레임 수
    folder_idx = 0     # 하위 폴더 인덱스 (0000, 0001, ...)
    frames_in_batch = 0  # 현재 폴더에 저장된 프레임 수

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            if frames_in_batch == 0 :
                batch_dir = os.path.join(output_dir, f"{folder_idx : 04d}")
                os.makedirs(batch_dir, exist_ok=True)

            frame_filename = os.path.join(output_dir, f"{saved:04d}.jpg")# 프레임 저장
            cv2.imwrite(frame_filename, frame)
            saved += 1
            frames_in_batch+=1

            if frames_in_batch ==10 :
                folder_idx +=1
                frames_in_batch = 0

        count += 1

    cap.release()
    print(f"✅ {os.path.basename(video_path)} → 총 {saved}장 저장됨 ({folder_idx + 1} 폴더 생성)")

def preprocess_dataset(dataset_link, save_dir, T=10, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )

    os.makedirs(save_dir, exist_ok=True)

    for i, (frame_folder, label) in enumerate(tqdm(dataset_link)):
        feature_save_path = os.path.join(save_dir, f"sample_{i}.pt")
        if os.path.exists(feature_save_path):
            continue  # 이미 전처리된 경우 건너뜀
        try:
            img_files = sorted([
                f for f in os.listdir(frame_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
        except FileNotFoundError:
            print(f"[SKIP] {frame_folder}: 경로를 찾을 수 없습니다")
            continue

        if len(img_files) < T:#frame이 10개보다 적은 경우 건너뜀
            print(f"[SKIP] {frame_folder}: insufficient frames")
            continue

        img_paths = [os.path.join(frame_folder, f) for f in img_files[:T]]

        faces = []
        last_valid_face = None

        for img_path in img_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"[ERROR] Failed to load image: {img_path}")
                continue

            face = extract_face(frame, face_mesh)
            if face is None:
                if last_valid_face is not None:
                    face = last_valid_face
                else:
                    print(f"[SKIP] {img_path}: face not detected")
                    break
            else:
                last_valid_face = face

            faces.append(face)

        if len(faces) < T:
            print(f"[SKIP] {frame_folder}: failed to get enough faces. {len(faces)}개.")
            continue

        # CNN feature 추출
        features = extract_cnn_features(faces, device) 
        try:
            torch.save({
                'features': features.cpu(),
                'label': torch.tensor([label], dtype=torch.float32)
            }, feature_save_path)
        except Exception as e:
            print(f"[ERROR] Saving failed: {e}")



if __name__ == "__main__":
    video_path = "sample.avi"
    output_dir = "../dataset2/train"
    frame_interval = calculate_frame_interval(video_path, target_fps=1) #1초에 1장
    extract_frames_from_video(video_path, output_dir, frame_interval)

    # 예시: 데이터셋 경로와 라벨 구성 (이 부분은 사용자가 정의)
    # dataset_link = [
    #     ("../dataset2/train/0000", 1),
    #     ("../dataset2/train/0001", 0),
    #     # ...
    # ]

    save_dir = "../dataset2/preprocessed_features/train"
    preprocess_dataset(dataset_link, save_dir)