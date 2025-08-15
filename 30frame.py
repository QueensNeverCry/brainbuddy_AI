import cv2
import os
import shutil
from glob import glob
from tqdm import tqdm

def extract_frames(video_path, local_output_base, segment_duration=10, target_fps=3, max_frames=30):
    """
    세그먼트당 정확히 30프레임 저장
    - target_fps=3 → 10초 구간에서 3fps × 10초 = 30프레임
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상 열기 실패 : {video_path}")
        return

    # 실제 FPS 읽고, 실패하면 30으로 가정
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_interval = max(int(round(fps / target_fps)), 1)
    segment_frame_count = int(round(segment_duration * fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_segments = total_frames // segment_frame_count

    print(f"🎬 {os.path.basename(video_path)} | 세그먼트 수: {num_segments}, "
          f"FPS={fps:.2f}, Interval={frame_interval}프레임마다 저장")

    for segment_idx in tqdm(range(num_segments), desc="세그먼트별 프레임 추출"):
        local_segment_dir = os.path.normpath(os.path.join(local_output_base, f"segment_{segment_idx}"))

        # 이미 max_frames 이상 있으면 스킵
        if os.path.exists(local_segment_dir):
            jpg_files = [f for f in os.listdir(local_segment_dir) if f.lower().endswith(".jpg")]
            if len(jpg_files) >= max_frames:
                print(f"✅ 세그먼트 {segment_idx} 이미 {len(jpg_files)}장 존재 → 건너뜀.")
                continue
            else:
                print(f"♻️ 세그먼트 {segment_idx} 프레임 {len(jpg_files)}장 → 삭제 후 재처리")
                shutil.rmtree(local_segment_dir)

        os.makedirs(local_segment_dir, exist_ok=True)

        # 세그먼트 시작 위치로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

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
                    print(f"⚠️ 세그먼트 {segment_idx} 프레임 읽기 실패, 재시도 {retry_count}/2... "
                          f"(프레임: {current_frame_pos})")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                    continue
                else:
                    print(f"❌ 세그먼트 {segment_idx} 프레임 읽기 실패 초과. 건너뜀.")
                    break

            retry_count = 0

            # interval에 맞춰 프레임 저장
            if count % frame_interval == 0:
                frame_path = os.path.normpath(os.path.join(local_segment_dir, f"{saved:04d}.jpg"))
                success = cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if success:
                    saved += 1
            count += 1

        print(f"✅ 세그먼트 {segment_idx} 저장 완료 ({saved}장)")

    cap.release()
    print("전체 작업 완료")


if __name__ == "__main__":
    # ✅ 입력(영상) 폴더
    video_folder = r"C:\Users\user\Pictures\Camera Roll"
    # ✅ 출력(프레임) 루트 폴더
    local_root = r"C:\f\camera_roll_frames"

    # mp4/avi/mov/m4v 등 확장자 지원
    exts = ("*.mp4", "*.avi", "*.mov", "*.m4v", "*.MP4", "*.AVI", "*.MOV", "*.M4V")
    video_files = []
    for ext in exts:
        video_files.extend(glob(os.path.join(video_folder, ext)))

    print(f"찾은 영상 수: {len(video_files)}개")

    for video_path in sorted(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        local_output_base = os.path.join(local_root, video_name)
        extract_frames(
            video_path,
            local_output_base,
            segment_duration=10,  # 10초 단위 세그먼트
            target_fps=3,         # 3fps로 캡처
            max_frames=30         # 세그먼트당 최대 30장
        )
