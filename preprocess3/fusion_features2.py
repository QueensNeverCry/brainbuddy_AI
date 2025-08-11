# -*- coding: utf-8 -*-
"""
AIHub 프레임 전처리 스크립트 (Thread/Process 선택형)
- 기본: ThreadPool (pickle 에러 회피, 가장 안정적)
- 옵션: --mode process (Windows에서 spawn + 안정화 설정)

사용 예:
python preprocess.py --base-dir "C:/AIhub_frames/test" --mode thread
python preprocess.py --base-dir "C:/AIhub_frames/test" --mode process
"""

import os
import cv2
import pickle
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

# mediapipe는 무거우므로 import 시점은 전역 유지하되, 객체 생성은 반드시 워커 내부에서!
import mediapipe as mp_solutions


# ===== 설정 기본값 =====
DEFAULT_BASE_DIR = r"C:/AIhub_frames/test"
FUSION_FEATURE_DIM = 5
IMAGE_SIZE = (640, 360)


# ===== FaceMesh 초기화 (워커 내부에서만 호출) =====
def init_face_mesh():
    # static_image_mode=True: 단일 프레임 처리에 적합
    return mp_solutions.solutions.face_mesh.FaceMesh(static_image_mode=True)


# ===== 개별 세그먼트 처리 =====
def process_segment(segment_path):
    """
    segment_path 내부에서 0001.jpg ... 등의 프레임 30장을 사용하여 특징을 추출하고
    fusion_features.pkl 로 저장. 문제 시 간단한 에러 문자열 반환, 정상 시 None 반환.
    """
    try:
        fusion_feat_path = os.path.join(segment_path, "fusion_features.pkl")
        if os.path.exists(fusion_feat_path):
            return None  # 이미 처리됨

        # 프레임 파일 수집
        try:
            files = os.listdir(segment_path)
        except Exception as e:
            return f"ERR|listdir|{segment_path}|{e}"

        image_files = sorted([
            f for f in files
            if f.lower().endswith(".jpg") and f[:4].isdigit()
        ])

        if len(image_files) < 30:
            return f"WARN|few_frames|{segment_path}|{len(image_files)}"

        image_files = image_files[:30]

        # 워커 안에서 FaceMesh 생성
        face_mesh = init_face_mesh()

        yaw_list, pitch_list, head_movement = [], [], []
        eye_closed_frames = 0
        yawn_detected = 0
        prev_nose = None

        for fname in image_files:
            img_path = os.path.join(segment_path, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark

            # 머리 이동 (코 포인트 기준)
            nose = landmarks[1]
            if prev_nose is not None:
                dx = nose.x - prev_nose.x
                dy = nose.y - prev_nose.y
                head_movement.append((dx * dx + dy * dy) ** 0.5)
            prev_nose = nose

            # 하품 (입술 거리)
            lip_dist = abs(landmarks[13].y - landmarks[14].y)
            if lip_dist > 0.05:
                yawn_detected = 1

            # 눈 감김 (윗/아랫눈꺼풀 사이)
            eye_openness = abs(landmarks[159].y - landmarks[145].y)
            if eye_openness < 0.015:
                eye_closed_frames += 1

            # 얼굴 각도 대용 특징 (좌우/상하 랜드마크 차)
            yaw = landmarks[263].x - landmarks[33].x
            pitch = landmarks[152].y - landmarks[10].y
            yaw_list.append(yaw)
            pitch_list.append(pitch)

        # 통계량 계산
        yaw_diff = np.std(np.diff(yaw_list)) if len(yaw_list) > 1 else 0.0
        pitch_diff = np.std(np.diff(pitch_list)) if len(pitch_list) > 1 else 0.0
        eye_closed_ratio = float(eye_closed_frames) / 30.0
        head_speed = float(np.mean(head_movement)) if head_movement else 0.0

        features = [float(yaw_diff), float(pitch_diff),
                    float(yawn_detected), float(eye_closed_ratio), float(head_speed)]

        if len(features) != FUSION_FEATURE_DIM:
            return f"ERR|feat_dim|{segment_path}|{len(features)}"

        # 결과 저장 (pickle)
        try:
            with open(fusion_feat_path, "wb") as f:
                pickle.dump(features, f)
        except Exception as e:
            return f"ERR|pickle_dump|{segment_path}|{e}"

        return None  # 성공

    except Exception as e:
        # 워커 내부 예외는 간단 문자열로 반환(이모지/특수문자 제외)
        return f"ERR|exception|{segment_path}|{e}"


# ===== 전체 세그먼트 경로 수집 =====
def get_all_segment_paths(base_dir):
    all_segments = []
    try:
        subjects = os.listdir(base_dir)
    except Exception:
        return all_segments

    for subject_folder in subjects:
        subject_path = os.path.join(base_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue
        try:
            segments = os.listdir(subject_path)
        except Exception:
            continue
        for segment_folder in segments:
            segment_path = os.path.join(subject_path, segment_folder)
            if os.path.isdir(segment_path):
                all_segments.append(segment_path)
    return all_segments


# ===== 실행 모드: ThreadPool =====
def run_threaded(segment_paths, workers):
    errors = []
    # ThreadPool 은 multiprocessing.dummy 사용
    from multiprocessing.dummy import Pool as ThreadPool

    with ThreadPool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(process_segment, segment_paths), total=len(segment_paths)):
            if result is not None:
                errors.append(result)
    return errors


# ===== 실행 모드: Process Pool (Windows 안정화) =====
def run_process(segment_paths, workers):
    errors = []

    # OpenCV 내부 스레드 비활성화(충돌 방지)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # Windows 기본 spawn을 명시, 작업자 수명 제한
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        # chunksize=1: 파이프 버퍼에 큰 덩어리 안 쌓이도록
        for result in tqdm(pool.imap_unordered(process_segment, segment_paths, chunksize=1),
                           total=len(segment_paths)):
            if result is not None:
                errors.append(result)
    return errors


def parse_args():
    p = argparse.ArgumentParser(description="AIHub 프레임 전처리 (MediaPipe FaceMesh 특징 추출)")
    p.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="세그먼트 루트 폴더")
    p.add_argument("--mode", type=str, choices=["thread", "process"], default="process",
                   help="실행 모드 선택 (기본 process)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2),
                   help="동시 작업자 수 (기본: CPU-1)")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = args.base_dir
    mode = args.mode
    workers = max(1, args.workers)

    print(f"📂 베이스 경로: {base_dir}")
    print(f"⚙️  실행 모드: {mode} | workers={workers}")

    segment_paths = get_all_segment_paths(base_dir)
    print(f"🧩 총 세그먼트 수: {len(segment_paths)}")

    if not segment_paths:
        print("세그먼트가 없습니다. --base-dir 경로를 확인하세요.")
        return

    if mode == "thread":
        errors = run_threaded(segment_paths, workers)
    else:
        errors = run_process(segment_paths, workers)

    print("\n✅ 전처리 완료!")
    if errors:
        print(f"\n⚠️ 오류/스킵 {len(errors)}개:")
        for e in errors:
            print(e)


if __name__ == "__main__":
    # Windows에서 multiprocessing 사용 시 필요 (특히 .exe 빌드나 IDLE 등)
    mp.freeze_support()
    main()
