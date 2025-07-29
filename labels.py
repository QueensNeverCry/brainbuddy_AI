import os
from collections import defaultdict, Counter
import pandas as pd

root_path = r"C:\Users\user\Downloads\126.디스플레이 중심 안구 움직임 영상 데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL"
target_displays = {"Monitor", "Laptop"}

# 사용자별 시나리오 개수, 조합 개수 저장용
user_scenario_counter = defaultdict(Counter)
user_combination_counter = defaultdict(Counter)

for user_id in os.listdir(root_path):
    user_path = os.path.join(root_path, user_id)
    if not os.path.isdir(user_path):
        continue
    for shot in os.listdir(user_path):
        shot_path = os.path.join(user_path, shot)
        if not os.path.isdir(shot_path):
            continue
        for display in os.listdir(shot_path):
            if display not in target_displays:
                continue
            json_path = os.path.join(shot_path, display, "json_rgb")
            if not os.path.isdir(json_path):
                continue
            for filename in os.listdir(json_path):
                if not filename.endswith(".json"):
                    continue
                parts = filename.split("_")
                try:
                    scenario = parts[4]   # 예: S03
                    state = parts[6]      # 예: S
                    pose = parts[7]       # 예: E
                    cam_dir = parts[8]    # 예: T
                except IndexError:
                    continue  # 예외 처리: 형식 안 맞는 파일 무시

                user_scenario_counter[user_id][scenario] += 1
                user_combination_counter[user_id][(display, state, pose, cam_dir)] += 1

# ▶ 사용자별 시나리오 DataFrame
scenario_df = pd.DataFrame(user_scenario_counter).fillna(0).astype(int).T
scenario_df.index.name = 'UserID'

# ▶ 사용자별 조합 DataFrame
comb_rows = []
for user, combs in user_combination_counter.items():
    for (disp, state, pose, cam), count in combs.items():
        comb_rows.append({
            "UserID": user,
            "디스플레이": disp,
            "상태": state,
            "자세": pose,
            "방향": cam,
            "개수": count
        })
comb_df = pd.DataFrame(comb_rows)
filtered_comb_df = comb_df[comb_df["자세"].isin(["D", "C"])]

# ▶ 콘솔 출력
# print("\n=== 사용자별 시나리오 개수 ===")
# print(scenario_df)

# print("\n=== 사용자별 조합별 개수 ===")
# print(comb_df)
# ▶ 콘솔 출력
print("\n=== 자세가 D 또는 C인 사용자별 조합별 개수 ===")
print(filtered_comb_df)

# ▶ CSV 파일로 저장
output_dir = "./output_csv"
os.makedirs(output_dir, exist_ok=True)

scenario_csv_path = os.path.join(output_dir, "사용자별_시나리오_개수.csv")
comb_csv_path = os.path.join(output_dir, "사용자별_조합별_개수.csv")

scenario_df.to_csv(scenario_csv_path, encoding='utf-8-sig')
filtered_comb_df.to_csv(comb_csv_path, index=False, encoding='utf-8-sig')

print(f"\n[저장 완료] 시나리오 요약: {scenario_csv_path}")
print(f"[저장 완료] 조합별 요약: {comb_csv_path}")


# ▶ 조합별 프레임 수 통계 출력
print("\n=== 조합별 프레임 수 통계 (상위 20개) ===")
scenario_count_df = (
    filtered_comb_df
    .groupby(["UserID", "디스플레이", "상태", "자세", "방향"])["개수"]
    .sum()
    .reset_index()
    .sort_values("개수", ascending=False)
)

print(scenario_count_df.head(20))

# ▶ 사용자별 조합 편차 확인용 예시 출력
print("\n=== 사용자별 조합별 프레임 수 요약 ===")
for user_id, group in scenario_count_df.groupby("UserID"):
    print(f"\n▶ 사용자: {user_id}")
    print(group.sort_values("개수", ascending=False).head(15))  # 상위 5개만 표시
