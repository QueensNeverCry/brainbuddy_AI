import cv2
import os
import shutil
from tqdm import tqdm
import mediapipe as mp
import models.face_crop3 as fc  # fc.crop_face, fc.last_face_bbox, fc.miss_count 를 사용
import random

def extract_frames(
    video_path, local_output_base, face_detector,
    # 5초당 15프레임(=3fps) 저장
    segment_duration=5, target_fps=3, max_frames=15,
    # on/off 샘플링 설정: 5초(on) 수집, 5초(off) 휴식 => on_segments=1, off_segments=1
    on_segments=1, off_segments=1, start_phase="random",
    # face crop 파라미터
    margin_x=0.2, margin_y=0.05, min_conf=0.6, miss_limit=10
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"영상 열기 실패 :{video_path}")
        return

    # 실제 FPS 사용 (fallback 30)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    fps = int(round(fps))

    frame_interval = max(int(fps / target_fps), 1)              # 3fps @ 30fps => 10프레임마다 1장
    segment_frame_count = int(round(segment_duration * fps))     # 5초 = 150프레임(30fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_segments = total_frames // segment_frame_count

    cycle = max(1, on_segments + off_segments)
    if start_phase == "random":
        phase0 = random.randint(0, cycle - 1)
    else:
        # 정수(0~cycle-1) 또는 0으로 고정
        phase0 = int(start_phase) % cycle

    print(
        f"🎬 세그먼트 수: {num_segments}, FPS:{fps}, Interval:{frame_interval}프레임마다 저장 | "
        f"On/Off: {on_segments}/{off_segments} (cycle={cycle}, start_phase={phase0})"
    )

    for segment_idx in tqdm(range(num_segments), desc=f"{segment_duration}초 단위(on/off)"):
        # on/off 판별: (segment_idx - phase0) % cycle < on_segments => 수집
        in_on_window = ((segment_idx - phase0) % cycle) < on_segments
        if not in_on_window:
            # off 구간이면 이 5초 블럭은 건너뜀
            continue

        local_segment_dir = os.path.normpath(os.path.join(local_output_base, f"segment_{segment_idx}"))
        fail_dir = f"../log/save_fail_img/{segment_idx}"

        # 이미 저장된 세그먼트 처리
        if os.path.exists(local_segment_dir):
            jpg_files = [f for f in os.listdir(local_segment_dir)
                         if f.lower().endswith(".jpg") and not f.startswith("fail_")]
            if len(jpg_files) >= max_frames:
                print(f"✅ 세그먼트 {segment_idx} 이미 {len(jpg_files)}장 존재 → 건너뜀.")
                continue
            else:
                print(f"♻️ 세그먼트 {segment_idx} 프레임 {len(jpg_files)}장 → 덮어쓰기 위해 삭제 후 재처리")
                shutil.rmtree(local_segment_dir)

        cap.set(cv2.CAP_PROP_POS_FRAMES, segment_idx * segment_frame_count)

        count = 0
        saved = 0
        retry_count = 0

        os.makedirs(local_segment_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)

        # 세그먼트 시작 시 상태 리셋(선택: 독립성↑)
        fc.last_face_bbox = None
        fc.miss_count = 0

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
                # face_crop3가 내부 고정 경로로 실패 저장한다면 save_fail_dir 인자 제거
                cropped = fc.crop_face(
                    frame, face_detector,
                    margin_x=margin_x, margin_y=margin_y,
                    min_conf=min_conf, miss_limit=miss_limit
                )
                if cropped is not None:
                    frame_path = os.path.normpath(os.path.join(local_segment_dir, f"{saved:04d}.jpg"))
                    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                    if cv2.imwrite(frame_path, cropped_bgr):
                        saved += 1
            count += 1

    cap.release()
    print("전체 작업 완료")



if __name__ == "__main__":
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    for i in range(21, 26):
        video_folder = f"C:/Users/user/Downloads/109.학습태도 및 성향 관찰 데이터/3.개방데이터/1.데이터/Training/01.원천데이터\TS_20_01_1/{i}"
        local_root = r"C:/AIhub_frames2/train"  # ✅ 로컬 저장 위치

        video_files = sorted([f for f in os.listdir(video_folder) if f.lower().endswith(".mp4")])

        for video_file in video_files:
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            local_output_base = os.path.join(local_root, video_name)

            # 🔹 세그먼트마다 좌우 넓게(귀/옆머리 포함), 상하는 적당히
            extract_frames(
                video_path, local_output_base, face_detector,
                segment_duration=5, target_fps=3, max_frames=15,  # 5초에 15장
                on_segments=1, off_segments=1,  # 5초 수집, 5초 휴식
                start_phase="random",           # 각 비디오마다 시작 위치 무작위 → 동기화 바이어스 완화
                margin_x=0.1, margin_y=0.05, min_conf=0.6, miss_limit=10
            )
        print(f"================={i}번 폴더 전처리 완료 ===================\n")

    face_detector.close()
