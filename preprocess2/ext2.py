import cv2
import os
import shutil
import time
from tqdm import tqdm

# 💡 안전하게 폴더 삭제 (파일 사용 중일 경우 재시도)
def safe_rmtree(path, retry=3, delay=1):
    for i in range(retry):
        try:
            shutil.rmtree(path)
            return
        except Exception as e:
            print(f"❗ 폴더 삭제 실패 ({i+1}/{retry}) → {e}")
            time.sleep(delay)
    print(f"🚨 삭제 포기: {path}")

def extract_frames(video_path, local_output_base, segment_duration=10, target_fps=10, max_frames=100):
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

    for segment_idx in tqdm(range(num_segments), desc="100프레임 단위로 분리"):
        local_segment_dir = os.path.normpath(os.path.join(local_output_base, f"segment_{segment_idx}"))

        if os.path.exists(local_segment_dir):
            jpg_files = [f for f in os.listdir(local_segment_dir) if f.lower().endswith(".jpg")]
            if len(jpg_files) >= max_frames:
                print(f"✅ 세그먼트 {segment_idx} 이미 {len(jpg_files)}장 존재 → 건너뜀.")
                continue
            else:
                print(f"♻️ 세그먼트 {segment_idx} 프레임 {len(jpg_files)}장 → 삭제 후 재처리")
                cap.release()  # 🔓 혹시나 영상이 해당 폴더를 잡고 있을 수 있음
                safe_rmtree(local_segment_dir)
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

        count = 0
        saved = 0
        retry_count = 0

        os.makedirs(local_segment_dir, exist_ok=True)

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

    cap.release()
    print("전체 작업 완료")

if __name__ == "__main__":
    for i in range(10, 16):
        video_folder = f"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Validation/01.원천데이터/vs_20/20_01/{i}"
        local_root = r"C:/f/valid/20_01"

        try:
            video_files = sorted([
                f for f in os.listdir(video_folder)
                if f.lower().endswith(".mp4")
            ])
        except FileNotFoundError:
            print(f"🚫 폴더 없음: {video_folder}")
            continue

        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            local_output_base = os.path.join(local_root, video_name)

            extract_frames(video_path, local_output_base)

        print(f"================={i}번 폴더 전처리 완료 ===================\n")
