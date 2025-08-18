import os
import csv
from collections import defaultdict

# 경로 설정
root_dir = r'C:\Users\user\Downloads\126.디스플레이 중심 안구 움직임 영상 데이터\01-1.정식개방데이터\Training\02.라벨링데이터\TL'
target_displays = ['Monitor', 'Laptop']
valid_conditions = {'S', 'F'}
valid_postures = {'C', 'D'}
valid_targets = {'T'}
group_keys = ['F_C', 'F_D', 'S_C', 'S_D']

# 결과 저장
summary_rows = []

# 인물 ID 순회
for person_id in sorted(os.listdir(root_dir)):
    person_path = os.path.join(root_dir, person_id)
    if not os.path.isdir(person_path):
        continue

    t1_path = os.path.join(person_path, 'T1')
    if not os.path.isdir(t1_path):
        continue

    for display in target_displays:
        json_rgb_path = os.path.join(t1_path, display, 'json_rgb')
        if not os.path.isdir(json_rgb_path):
            continue

        json_files = [f for f in os.listdir(json_rgb_path) if f.endswith('.json')]

        group_sum = {key: 0 for key in group_keys}

        for fname in json_files:
            parts = fname.replace('.json', '').split('_')
            if len(parts) < 11:
                continue

            condition = parts[7]
            posture = parts[8]
            target = parts[9]

            if (
                condition in valid_conditions and
                posture in valid_postures and
                target in valid_targets
            ):
                group_key = f"{condition}_{posture}"
                if group_key in group_sum:
                    group_sum[group_key] += 1

        # 하나라도 조건에 맞는 게 있으면 기록
        if any(count > 0 for count in group_sum.values()):
            summary_rows.append({
                'ID': person_id,
                'Display': display,
                'F_C_T': group_sum['F_C'],
                'F_D_T': group_sum['F_D'],
                'S_C_T': group_sum['S_C'],
                'S_D_T': group_sum['S_D'],
                'Target': 'T'
            })

# CSV 저장
output_csv = 'filtered_summary_T_only.csv'
with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['ID', 'Display', 'F_C_T', 'F_D_T', 'S_C_T', 'S_D_T', 'Target'])
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"✅ CSV 저장 완료: {output_csv}")
    