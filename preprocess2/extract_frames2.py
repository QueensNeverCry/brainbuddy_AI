import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_base_path, segment_duration=10, target_fps=30, max_frames=100):
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

        output_dir = os.path.join(output_base_path, str(segment_idx))
        os.makedirs(output_dir, exist_ok=True)

        count = 0
        saved = 0
        retry_count = 0

        while saved < max_frames:
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame_pos >= (segment_idx + 1) * segment_frame_count:
                break

            ret, frame = cap.read()
            if not ret:
                if retry_count < 2:
                    retry_count += 1
                    print(f"⚠️ 세그먼트 {segment_idx} 프레임 읽기 실패, 재시도 {retry_count}/2...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    continue
                else:
                    print(f"❌ 세그먼트 {segment_idx} 프레임 읽기 실패 초과. 해당 세그먼트 건너뜀.")
                    break

            retry_count = 0

            if count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{saved:04d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved += 1

            count += 1

        print(f"📁 세그먼트 {segment_idx} 완료 ({saved}장 저장)")

    cap.release()
    print("전체 작업 완료")

if __name__ == "__main__":
    for i in range(11, 21):  # 1부터 10까지
        video_folder = f"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Validation/01.원천데이터/vs_20/20_03/{i}"
        output_root = "C:/f/valid/20_03"

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
            output_base_path = os.path.join(output_root, video_name)

            extract_frames(video_path, output_base_path)

        print(f"================= ✅ {i}번 폴더 전처리 완료 ===================\n")
