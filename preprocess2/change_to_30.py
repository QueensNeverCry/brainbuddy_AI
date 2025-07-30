import os
import glob
from tqdm import tqdm

def keep_evenly_sampled_images(root_dir, keep_count=30):
    segment_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.lower().endswith(".jpg") for f in filenames):
            segment_dirs.append(dirpath)

    for segment_dir in tqdm(segment_dirs, desc="📂 세그먼트 정리 중"):
        jpg_files = sorted(
            glob.glob(os.path.join(segment_dir, "*.jpg"))
        )
        total = len(jpg_files)
        if total <= keep_count:
            continue

        # 균등 간격 인덱스 계산
        interval = total / keep_count
        keep_indices = [round(i * interval) for i in range(keep_count)]
        keep_indices = sorted(set(min(i, total - 1) for i in keep_indices))  # 범위 초과 방지

        keep_files = [jpg_files[i] for i in keep_indices]
        delete_files = [f for f in jpg_files if f not in keep_files]

        for f in delete_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"❌ 삭제 실패: {f} → {e}")

        print(f"📁 {segment_dir}: {total}장 중 {len(keep_files)}장 유지, {len(delete_files)}장 삭제")

if __name__ == "__main__":
    local_root = r"C:/AIhub_frames/train"  # ✅ 이미지 저장 루트
    keep_evenly_sampled_images(local_root, keep_count=30)
