# clip1_4.py
# ------------------------------------------------------------
# Step 4: 학습된 Linear Probe로 전체 폴더 예측 → labels_final.csv 저장
# - embeddings.npz: X(임베딩), ids(폴더경로)
# - linear_probe.npz: W, b (로지스틱 이진)
# - probe_meta.json: (선택) 기타 메타 — engine 키 없어도 동작
# ------------------------------------------------------------

import os, json
import numpy as np
import pandas as pd
from datetime import datetime

EMB_PATH = "embeddings.npz"
PROBE_PATH = "linear_probe.npz"
META_PATH = "probe_meta.json"   # 없어도 무방
OUT_CSV  = "labels_final.csv"

# 최종 임계값(요청값)
THRESHOLD = 0.700

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def main():
    # 1) 파일 체크
    for p in [EMB_PATH, PROBE_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    # 2) 임베딩 로드 (X, ids)
    emb = np.load(EMB_PATH, allow_pickle=True)
    if not {"X", "ids"}.issubset(set(emb.files)):
        raise KeyError(f"{EMB_PATH} must contain 'X' and 'ids' keys, found: {emb.files}")
    X   = emb["X"]      # shape: (N, D)
    ids = emb["ids"]    # shape: (N,)  - 폴더 경로 문자열 배열
    N, D = X.shape
    print(f"[Load] embeddings: N={N}, D={D}")

    # 3) 프로브 로드 (W, b)
    probe = np.load(PROBE_PATH, allow_pickle=True)
    if not {"W", "b"}.issubset(set(probe.files)):
        raise KeyError(f"{PROBE_PATH} must contain 'W' and 'b' keys, found: {probe.files}")
    W = probe["W"].reshape(1, -1)     # (1, D)
    b = probe["b"].reshape(1,)        # (1,)
    if W.shape[1] != D:
        raise ValueError(f"Dim mismatch: W has {W.shape[1]} dims but X has {D}")

    # (선택) 엔진 정보는 probe 파일에서만 시도해서 읽고, 실패 시 unknown
    try:
        model_name = probe["MODEL_NAME"].item()
        pretrained = probe["PRETRAINED"].item()
        engine = f"{model_name}/{pretrained}"
    except Exception:
        engine = "unknown"

    # (선택) meta 파일이 있어도 'engine'은 참조하지 않음 (없어도 OK)
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    else:
        meta = {}

    print(f"[Meta] threshold={THRESHOLD:.3f}  engine={engine}")

    # 4) 예측 (p_positive는 학습에서 1로 둔 클래스의 확률)
    z = X @ W.T + b            # shape: (N, 1)
    p = sigmoid(z).reshape(-1) # (N,)
    pred = (p >= THRESHOLD).astype(int)

    # 5) 저장
    df = pd.DataFrame({
        "folder": ids,
        "p_positive": p,                # 1(집중) 확률(정의에 맞게 해석)
        "predicted_label": pred,        # 1=집중, 0=비집중
        "threshold": THRESHOLD,
        "engine": engine,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved] {OUT_CSV} (rows={len(df)})")

if __name__ == "__main__":
    main()
