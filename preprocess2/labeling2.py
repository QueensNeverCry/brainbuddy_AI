
import os
import pickle
from pathlib import Path
import pandas as pd

# -------------------- USER SETTINGS --------------------
# 이미지 루트 폴더 (아래에 001, 002 등 하위 폴더가 있고, 그 아래에 frame 폴더들이 있음)
BASE_ROOT = r"C:/AIhub_eye_frames"
# CSV 경로 (folder, predicted_label 컬럼 필요)
CSV_PATH = r"C:/Users/user/Downloads/labels_final.csv"
# 생성할 PKL 경로
OUT_PKL  = r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/test/another_data.pkl"

# 프레임 수 요건
NUM_FRAMES_REQUIRED = 30
# 허용되는 이미지 확장자
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def count_frames(dirpath: Path) -> int:
    if not dirpath.is_dir():
        return 0
    cnt = 0
    for name in os.listdir(dirpath):
        if Path(name).suffix.lower() in IMG_EXTS:
            cnt += 1
    return cnt


def index_all_leaf_dirs(root: Path) -> dict:
    """
    root 하위 모든 디렉터리를 순회하며 leaf 폴더명을 key로 하고,
    해당 경로 목록(Path 리스트)을 value로 하는 dict 반환.
    같은 leaf 이름이 여러 위치에 있을 수 있으므로 리스트로 관리.
    """
    mapping = {}
    for cur, dirs, files in os.walk(root):
        # leaf 디렉터리만 관심
        if not dirs:  # 더 하위 폴더가 없으면 leaf로 간주
            leaf = Path(cur).name
            mapping.setdefault(leaf, []).append(Path(cur))
    return mapping


def main():
    base_root = Path(BASE_ROOT)
    if not base_root.exists():
        raise FileNotFoundError(f"BASE_ROOT not found: {base_root}")

    df = pd.read_csv(CSV_PATH)
    if "folder" not in df.columns or "predicted_label" not in df.columns:
        raise ValueError("CSV must contain columns: 'folder', 'predicted_label'")

    # leaf 이름만 추출 (CSV folder 컬럼이 전체 경로여도 OK)
    df["leaf"] = df["folder"].astype(str).apply(lambda p: Path(p).name)
    # 중복 제거(같은 leaf가 여러 번 나오면 마지막 행 우선, 필요시 규칙 조정)
    df = df.drop_duplicates(subset=["leaf"], keep="last")

    # 파일시스템에서 leaf -> 실제 경로 목록 인덱싱
    leaf_map = index_all_leaf_dirs(base_root)

    pairs = []  # (folder_path:str, label:int)
    missing, multi, short = 0, 0, 0

    for _, row in df.iterrows():
        leaf = row["leaf"]
        label = int(row["predicted_label"])
        candidates = leaf_map.get(leaf, [])

        # 파일시스템에 아예 없으면 스킵
        if not candidates:
            missing += 1
            continue

        # 후보 중 프레임 수 정확히 맞는 폴더만
        exact = [p for p in candidates if count_frames(p) == NUM_FRAMES_REQUIRED]
        if not exact:
            # 없으면 스킵
            short += 1
            continue

        # 여러 개면 첫 번째 사용 (원하면 우선순위 로직 추가 가능)
        if len(exact) > 1:
            multi += 1
        chosen = exact[0]

        pairs.append((str(chosen), label))

    # 저장
    out_path = Path(OUT_PKL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pairs, f)

    print(f"Saved: {out_path}  (#pairs={len(pairs)})")
    print(f"  missing leafs in FS: {missing}")
    print(f"  multiple matches resolved: {multi}")
    print(f"  folders with <{NUM_FRAMES_REQUIRED} frames: {short}")


if __name__ == "__main__":
    main()
