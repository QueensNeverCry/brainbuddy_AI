# probe_group_regex.py
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--regex", type=str, required=True)
parser.add_argument("--csv", type=str, default="labels_final.csv")
args = parser.parse_args()

df = pd.read_csv(args.csv)
pattern = re.compile(args.regex)

groups = []
for folder in df['folder']:
    m = pattern.search(folder)
    if m:
        groups.append(m.group(1))
    else:
        groups.append("NO_MATCH")

df['group_id'] = groups
unique_groups = df['group_id'].unique()

print(f"[INFO] Unique group count: {len(unique_groups)}")
print(f"[INFO] Example groups: {unique_groups[:10]}")
print(df['group_id'].value_counts())
