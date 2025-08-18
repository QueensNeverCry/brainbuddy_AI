# 전처리
# 프레임 개수를 100개로 통일
# 70개 이상~ 100개 이하 : 앞 뒤로 프레임을 균등한 개수로 붙이기
# 100개 이상 : 앞의 100개만 사용
# 70개 이하 : 사용하지 않음
import numpy as np

def normalize_frame_sequence_np(frames: np.ndarray, target_len=100, min_len=70) -> np.ndarray | None:
    """
    프레임 시퀀스를 NumPy 배열로 받아 100개로 보정하여 반환.

    Parameters:
    - frames: np.ndarray, shape = (N, H, W, C)
    - target_len: 최종 프레임 수
    - min_len: 최소 허용 프레임 수

    Returns:
    - shape = (100, H, W, C)
    - 프레임 수 부족 시 None
    """

    current_len = frames.shape[0]

    if current_len < min_len:
        return None  # 사용하지 않음

    elif current_len >= target_len:
        return frames[:target_len]  # 앞 100개만 사용

    else:
        # 앞뒤 균등 패딩
        pad_total = target_len - current_len
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front

        front_pad = np.repeat(frames[0:1], pad_front, axis=0)   # 첫 프레임 반복
        back_pad = np.repeat(frames[-1:], pad_back, axis=0)     # 마지막 프레임 반복

        padded = np.concatenate([front_pad, frames, back_pad], axis=0)
        return padded
