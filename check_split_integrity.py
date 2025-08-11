# check_split_integrity.py
import json, argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

def extract_group_from_path(folder: str, mode: str, depth: int, regex: str):
    p = Path(folder)
    parts = p.parts
    if mode == "regex":
        m = re.search(regex, folder.replace("\\", "/"))
        return m.group(1) if m else "UNKNOWN"
    if mode == "two_up":
        return parts[-3] if len(parts) >= 3 else parts[-1]
    if mode == "depth":
        cut = len(parts) - depth
        cut = max(1, cut)
        return "/".join(parts[max(0, cut-1):cut])
    # default: pair_up
    if len(parts) >= 4: return f"{parts[-4]}/{parts[-3]}"
    if len(parts) >= 3: return f"{parts[-3]}/{parts[-2]}"
    return parts[-1]

def counts(y_idx):
    ysum = int(y_idx.sum())
    return {"n": int(len(y_idx)), "pos": ysum, "neg": int(len(y_idx)-ysum)}

def main(args):
    df = pd.read_csv(args.csv)
    assert {"folder","predicted_label"}.issubset(df.columns), "CSV needs folder,predicted_label"

    # 그룹 산출
    groups = np.array([
        extract_group_from_path(str(p), args.group_mode, args.group_depth, args.group_regex)
        for p in df["folder"].tolist()
    ])
    y = df["predicted_label"].astype(int).to_numpy()

    # splits 로드 (clip2_1.py 저장 형식과 동일)
    meta = json.load(open(args.splits, "r", encoding="utf-8"))
    tr_idx = np.array(meta["train_indices"], dtype=int)
    va_idx = np.array(meta["val_indices"], dtype=int)
    te_idx = np.array(meta["test_indices"], dtype=int)

    # 각 split 그룹/분포
    g_train = set(groups[tr_idx]); g_val = set(groups[va_idx]); g_test = set(groups[te_idx])
    c_train = counts(y[tr_idx]); c_val = counts(y[va_idx]); c_test = counts(y[te_idx])

    # 교집합(누수) 체크
    leak_tv = sorted(list(g_train & g_val))
    leak_tt = sorted(list(g_train & g_test))
    leak_vt = sorted(list(g_val & g_test))
    leak_all = sorted(list(g_train & g_val & g_test))

    def has_both(idx):
        return (y[idx].sum() > 0) and (y[idx].sum() < len(idx))

    print("=== SPLIT INTEGRITY REPORT ===")
    print(f"Train: {c_train} | groups={len(g_train)} | both_classes={has_both(tr_idx)}")
    print(f"Val  : {c_val} | groups={len(g_val)} | both_classes={has_both(va_idx)}")
    print(f"Test : {c_test} | groups={len(g_test)} | both_classes={has_both(te_idx)}")

    warn = False
    if leak_tv or leak_tt or leak_vt:
        warn = True
        print("\n[WARN] Group overlap detected!")
        if leak_tv: print(f"- Train ∩ Val : {len(leak_tv)} groups")
        if leak_tt: print(f"- Train ∩ Test: {len(leak_tt)} groups")
        if leak_vt: print(f"- Val ∩ Test  : {len(leak_vt)} groups")
        if leak_all: print(f"- Train ∩ Val ∩ Test: {len(leak_all)} groups")
    else:
        print("\n[OK] No group overlap among splits.")

    # CSV 리포트 저장
    if args.out_report:
        rows = []
        for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
            gp = groups[idx]
            ys = y[idx]
            rows.append({
                "split": name,
                "n": len(idx),
                "pos": int(ys.sum()),
                "neg": int(len(ys)-ys.sum()),
                "n_groups": len(set(gp)),
                "both_classes": has_both(idx),
            })
        # 교집합 일부도 저장
        rep = pd.DataFrame(rows)
        rep.to_csv(args.out_report, index=False, encoding="utf-8-sig")
        if args.out_groups:
            pd.DataFrame({
                "train_only": sorted(list(g_train - g_val - g_test)),
                "val_only":   sorted(list(g_val - g_train - g_test)),
                "test_only":  sorted(list(g_test - g_train - g_val)),
            }).to_csv(args.out_groups, index=False, encoding="utf-8-sig")
        if args.out_overlaps:
            pd.DataFrame({
                "train_val": leak_tv,
                "train_test": leak_tt,
                "val_test": leak_vt,
                "all_three": leak_all
            }).to_csv(args.out_overlaps, index=False, encoding="utf-8-sig")
        print(f"\n[Saved] {args.out_report}"
              f"{' | '+args.out_groups if args.out_groups else ''}"
              f"{' | '+args.out_overlaps if args.out_overlaps else ''}")

    if warn:
        print("\n=> ACTION: 그룹 정의(--group-mode/--group-regex) 재검토 후 재스플릿 권장.")
    print("===============================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="labels_final.csv")
    ap.add_argument("--splits", type=str, default="splits_indices.json")
    ap.add_argument("--group-mode", type=str, default="pair_up",
                    choices=["pair_up","two_up","depth","regex"])
    ap.add_argument("--group-depth", type=int, default=3)
    ap.add_argument("--group-regex", type=str, default=r"/TS/(\d{3})/")
    ap.add_argument("--out-report", type=str, default="split_report.csv")
    ap.add_argument("--out-groups", type=str, default="split_groups.csv")
    ap.add_argument("--out-overlaps", type=str, default="split_overlaps.csv")
    args = ap.parse_args()
    main(args)
