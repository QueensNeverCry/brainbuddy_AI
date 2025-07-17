import cv2
import os
import shutil
from tqdm import tqdm

def extract_frames(video_path, local_output_base, drive_output_base, segment_duration=10, target_fps=30, max_frames=300):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"영상 열기 실패 :{video_path}")
        return

    fps = 30
    frame_interval = max(int(fps / target_fps), 1)
    segment_frame_count = segment_duration * fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_segments = total_frames // segment_frame_count

    print(f"🎬 세그먼트 수: {num_segments}, Interval: {frame_interval}프레임마다 저장")
    print(f"▶️ {os.path.basename(video_path)} 총 프레임 수: {total_frames}")

    for segment_idx in tqdm(range(num_segments), desc="300프레임 단위로 분리"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

        count = 0
        saved = 0
        retry_count = 0

        # ✅ 세그먼트별 로컬 저장 경로
        local_segment_dir = os.path.normpath(os.path.join(local_output_base, f"segment_{segment_idx}"))
        os.makedirs(local_segment_dir, exist_ok=True)

        print(f"📁 세그먼트 {segment_idx} 저장 시작 → {local_segment_dir}")

        while saved < max_frames:
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame_pos >= (segment_idx + 1) * segment_frame_count:
                break

            ret, frame = cap.read()
            if not ret:
                if retry_count < 2:
                    retry_count += 1
                    print(f"⚠️ 세그먼트 {segment_idx} 프레임 읽기 실패, 재시도 {retry_count}/2... (프레임: {current_frame_pos})")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    continue
                else:
                    print(f"❌ 세그먼트 {segment_idx} 프레임 읽기 실패 초과. 건너뜀.")
                    break

            retry_count = 0

            if count % frame_interval == 0:
                frame_path = os.path.normpath(os.path.join(local_segment_dir, f"{saved:04d}.jpg"))

                if frame is None:
                    print(f"❗ 프레임이 None입니다 (세그먼트 {segment_idx}, count {count})")
                else:
                    success = cv2.imwrite(frame_path, frame)
                    if not success:
                        print(f"❌ 프레임 저장 실패: {frame_path}")
                    else:
                        saved += 1

            count += 1

        print(f"✅ 세그먼트 {segment_idx} 저장 완료 ({saved}장)")

        # ✅ 세그먼트별 Google Drive 경로로 복사
        drive_segment_dir = os.path.normpath(os.path.join(drive_output_base, f"segment_{segment_idx}"))
        try:
            shutil.copytree(local_segment_dir, drive_segment_dir, dirs_exist_ok=True)
            print(f"📤 세그먼트 {segment_idx} → Google Drive 복사 완료: {drive_segment_dir}")

            # ✅ 복사 성공 시, 로컬 세그먼트 폴더 삭제
            shutil.rmtree(local_segment_dir)
            print(f"🧹 로컬 세그먼트 폴더 삭제 완료: {local_segment_dir}")

        except Exception as e:
            print(f"❌ Google Drive 복사 실패 (세그먼트 {segment_idx}): {e}")
            
    cap.release()
    print("전체 작업 완료")


if __name__ == "__main__":
    video_folder = r"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_20_01_2"
    local_root = r"C:/Temp/train_frames"  # ✅ 로컬 저장 위치
    drive_root_base = r"G:/내 드라이브/train/20_01"  # ✅ Google Drive 대상 경로

    video_files = sorted([
        f for f in os.listdir(video_folder)
        if f.lower().endswith(".mp4")
    ])

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]

        local_output_base = os.path.join(local_root, video_name)
        drive_output_base = os.path.join(drive_root_base, video_name)

        extract_frames(video_path, local_output_base, drive_output_base)
        print(f"================= {video_file} 전처리 완료 ===================\n")
