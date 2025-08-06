import os
import shutil
from collections import defaultdict

# ────────────────────────────────────────────────────────
# 1) “큰” 최상위 폴더 지정
big_base_root = r"C:/Users/user/Downloads/126.eye/01-2.data/training/Training/ogdata/TS/TS"

# 2) 결과를 저장할 최상위 폴더 (원본 옆에 생성)
output_base = os.path.join(big_base_root, "분류된_이미지_30개_전체")
os.makedirs(output_base, exist_ok=True)

# 3) 그룹 하나당 최대 개수
batch_size = 30

# ────────────────────────────────────────────────────────
# 4) 세션별(숫자 폴더별)로 처리
for session_name in sorted(os.listdir(big_base_root)):
    session_dir = os.path.join(big_base_root, session_name)
    if not os.path.isdir(session_dir):
        continue

    # 4-a) 해당 세션 안의 모든 'rgb' 폴더 찾기 → { base_name: [(fn, 폴더), …] }
    grouped = defaultdict(list)
    for root, dirs, files in os.walk(session_dir):
        if os.path.basename(root).lower() == "rgb":
            for fn in sorted(files):
                if fn.lower().endswith(".png"):
                    base = fn.rsplit("_", 1)[0]
                    grouped[base].append((fn, root))

    # 4-b) 그룹별로 폴더 만들고 이동 + 패딩 복제
    session_out = os.path.join(output_base, session_name)
    for base_name, entries in grouped.items():
        # 프레임 순서대로
        entries.sort(key=lambda x: x[0])

        dest_dir = os.path.join(session_out, base_name)
        os.makedirs(dest_dir, exist_ok=True)

        # 이동할 목록 결정
        if len(entries) >= batch_size:
            to_move = entries[:batch_size]
            pad_count = 0
        else:
            to_move = entries
            pad_count = batch_size - len(entries)

        # 원본 이동
        for fn, src_dir in to_move:
            shutil.move(os.path.join(src_dir, fn),
                        os.path.join(dest_dir, fn))

        # 부족분은 마지막 프레임 복제
        if pad_count > 0:
            last_fn, _ = entries[-1]
            last_moved = os.path.join(dest_dir, last_fn)
            name_no_ext, ext = os.path.splitext(last_fn)
            for i in range(pad_count):
                pad_name = f"{name_no_ext}_pad_{i+1:02d}{ext}"
                shutil.copy(last_moved,
                            os.path.join(dest_dir, pad_name))

# ────────────────────────────────────────────────────────
# 5) 완료 메시지
print("모든 세션 처리 완료!")
print("결과 폴더:", output_base)
