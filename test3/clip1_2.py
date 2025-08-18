# clip1_2.py
# ------------------------------------------------------------
# Step 2: 제로샷 결과 품질 점검 & 신뢰 샘플 선별 (분포 기반 기준)
# - labels_zeroshot.csv 로드 (경로 고정)
# - 요약 통계 / 분포 출력
# - |margin|의 분포(분위수)에 따라 high / mid / low CSV 저장
# - 의사 GT(sign(margin))로 threshold 근사 추천 + JSON 저장
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd

# ====== 경로 고정 ======
CSV_PATH = r"C:\Users\user\Desktop\brainbuddy_AI\labels_zeroshot.csv"

# ====== 파라미터 ======
# 분위수 기준 (데이터 분포 자동 적응)
HIGH_Q = 0.90    # |margin|의 상위 10% → high
MID_Q  = 0.60    # |margin|의 상위 40% → mid
TARGET_RECALL = 0.90
SUGGEST_THRESHOLD = True

REQUIRED_COLS = ["folder", "p_focused", "margin", "predicted_label"]

def check_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}\nCSV 컬럼을 확인하세요.")

def summarize(df: pd.DataFrame):
    print("=== Basic Summary ===")
    print(df.describe(include="all"))
    print("\n=== Counts by predicted_label ===")
    if "predicted_label" in df.columns:
        print(df["predicted_label"].value_counts(dropna=False))
    print("\n=== Margin describe ===")
    print(df["margin"].describe())

def quantile_hist(series: pd.Series, k: int = 12):
    s = series.dropna()
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(s, qs)
    edges = np.unique(edges)
    bins = pd.cut(s, bins=edges, include_lowest=True)
    vc = bins.value_counts().sort_index()
    return vc

def split_by_quantiles(df: pd.DataFrame, high_q: float, mid_q: float):
    abs_m = df["margin"].abs()
    high_thr = float(np.quantile(abs_m, high_q))
    mid_thr  = float(np.quantile(abs_m, mid_q))
    print(f"\n[Auto thresholds] |margin| mid_thr={mid_thr:.6f}, high_thr={high_thr:.6f}")

    hi = df[abs_m >= high_thr].copy()
    md = df[(abs_m >= mid_thr) & (abs_m < high_thr)].copy()
    lo = df[abs_m < mid_thr].copy()
    return hi, md, lo, high_thr, mid_thr

def save_splits(hi: pd.DataFrame, md: pd.DataFrame, lo: pd.DataFrame):
    hi.to_csv("labels_high_conf.csv", index=False, encoding="utf-8-sig")
    md.to_csv("labels_mid_conf.csv", index=False, encoding="utf-8-sig")
    lo.to_csv("labels_low_conf.csv", index=False, encoding="utf-8-sig")
    print("\nSaved:")
    print(f" - labels_high_conf.csv  (rows={len(hi)})")
    print(f" - labels_mid_conf.csv   (rows={len(md)})")
    print(f" - labels_low_conf.csv   (rows={len(lo)})")

def suggest_threshold(df: pd.DataFrame, target_recall: float = 0.90):
    """
    의사 GT(= sign(margin))로 전체에서 근사 평가.
    - margin > 0 → focused(1)
    - margin < 0 → unfocused(0)
    p_focused threshold를 스윕해 목표 리콜을 만족하는 값 추천.
    """
    work = df.dropna(subset=["p_focused","margin"]).copy()
    if len(work) == 0:
        return None, None, None

    y_true = (work["margin"] > 0).astype(int).values
    p = work["p_focused"].astype(float).values

    # p_focused 분포에 맞춰 조밀 스캔
    grid = np.linspace(max(0.05, p.min()), min(0.95, p.max()), 181)
    best = None
    for th in grid:
        pred = (p >= th).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        # 근사 precision/recall
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        score = (recall >= target_recall, precision, recall)
        if best is None or score > best[0]:
            best = (score, th, recall, precision)

    if best is None:
        return None, None, None
    _, th, r, p_ = best
    return float(th), float(r), float(p_)

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV를 찾을 수 없습니다: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    check_columns(df)
    summarize(df)

    print("\n=== Margin histogram (quantile bins) ===")
    print(quantile_hist(df["margin"], k=12))

    hi, md, lo, high_thr, mid_thr = split_by_quantiles(df, HIGH_Q, MID_Q)
    print(f"\nSplit counts → high(|m|>={high_thr:.6f}): {len(hi)}, "
          f"mid([{mid_thr:.6f},{high_thr:.6f})): {len(md)}, low(<{mid_thr:.6f}): {len(lo)}")
    save_splits(hi, md, lo)

    if SUGGEST_THRESHOLD:
        th, r, p_ = suggest_threshold(df, target_recall=TARGET_RECALL)
        if th is None:
            print("\n[Suggest] threshold 추천에 실패했습니다.")
        else:
            print(f"\n[Suggest] threshold ≈ {th:.3f}  (approx recall={r:.3f}, precision={p_:.3f})")
            with open("threshold_suggestion.json", "w", encoding="utf-8") as f:
                json.dump({
                    "suggested_threshold": th,
                    "approx_recall": r,
                    "approx_precision": p_,
                    "auto_margins": {
                        "mid_quantile": MID_Q, "mid_thr": mid_thr,
                        "high_quantile": HIGH_Q, "high_thr": high_thr
                    }
                }, f, ensure_ascii=False, indent=2)
            print("Saved: threshold_suggestion.json")

if __name__ == "__main__":
    main()
