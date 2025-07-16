import os
import json
import pickle
from pathlib import Path

def load_json_files(label_subdir):
    """라벨 서브디렉토리(예: label/60/01) 내의 모든 .json 파일 경로 리스트 반환"""
    label_subdir = Path(label_subdir)
    return list(label_subdir.glob("*.json")) # rglob : 하위폴더까지 재귀적으로 탐색

def extract_label_and_path(json_path, frame_root):
    """
    json 파일에서 (프레임 경로, 라벨) 추출
    :return: (frame_path, label) or None
    """
    try:
        json_path=Path(json_path)
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        category_name = data["이미지"]["category"]["name"]
        normalized = category_name.replace(" ", "")
        if normalized == "집중":
            label = 1
        elif normalized == "집중하지않음":
            label = 0
        else:
            label = None

        if label is None:
            print(f"[스킵됨] 알 수 없는 라벨: {category_name} ({json_path})")
            return None

        parts = json_path.parts # (앞의 경로..,'label', '60', '01','jsonfilename.json')
        idx = parts.index("label") + 1 
        part1 = parts[idx]        # '60'
        part2 = parts[idx + 1]    # '01', '02', ...
        last_number = json_path.stem.split("-")[-1]  # '0'

        # frame_root + 60_01 + 마지막 번호
        frame_path = Path(frame_root)/f"{part1}_{part2}"/last_number
        
        if not frame_path.exists():
            print(f"[경고] 프레임 폴더 없음: {frame_path}")
            return None

        return (frame_path, label)

    except Exception as e:
        print(f"[에러] 처리 중 예외 발생: {e} ({json_path})")
        return None

def save_pickle(data, save_path):
    """(path, label) 리스트를 pickle로 저장"""
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n✅ 총 {len(data)}개 경로를 저장했습니다.")
    print(f"📁 Pickle 경로: {save_path}")

def main():
    base_dir = "C:/GitHub/brainbuddy/preprocess2/AIHub_frames/train"
    label_base = "C:/GitHub/brainbuddy/AIHub/train/label"
    frame_base = base_dir
    pickle_save_path = "AIHub_label_mapping.pkl"

    # AIHub_frames 데이터셋 분류명 : 00_01 ~ 60_05
    target_sets = [
        "00_01", "00_02", "00_03", "00_04", "00_05",
        "60_01", "60_02", "60_03"
    ]

    video_label_pairs = []

    for set_name in target_sets:
        part1, part2 = set_name.split("_")  # ex: "60", "01"
        label_subdir = Path(label_base) / part1 / part2 # .. AIHub/train/label/60/01

        if not os.path.exists(label_subdir):
            print(f"[건너뜀] 라벨 폴더 없음: {label_subdir}")
            continue

        json_files = load_json_files(label_subdir)
        print(f"📂 {set_name} 내 JSON 파일 수: {len(json_files)}")

        for json_path in json_files:
            result = extract_label_and_path(json_path, frame_base)
            if result:
                video_label_pairs.append(result)
                
    print("\n=== 샘플 5개 출력 ===")
    for i, (path, label) in enumerate(video_label_pairs[:5]):
        print(f"{i+1}. 경로: {path}, 라벨: {label}")
        
    save_pickle(video_label_pairs, pickle_save_path)

if __name__ == "__main__":
    main()

