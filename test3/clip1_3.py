# clip1_3.py
# ------------------------------------------------------------
# Step 3: CLIP 임베딩 + Linear Probe 학습 (누수 방지: 그룹 분할)
# - embeddings.npz 재사용(없으면 생성)
# - labels_zeroshot.csv에서 극단 분위수로 pos/neg 선택(부호 무시)
# - GroupShuffleSplit으로 '사람/세션' 단위 그룹 분할 → 누수 방지
# - 검증셋에서 목표 리콜 맞춰 threshold 튜닝
# - 저장물: linear_probe.npz, probe_meta.json
# ------------------------------------------------------------

import os, json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

# ====== 경로 고정 ======
ROOT_DIRS = [
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/134_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/135_face_crop",
    r"C:/Users/user/Downloads/126.eye/0801/t/o/og/TS/TS/all_image_30/136_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/001/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/002/T1/image_30_face_crop",
    r"C:/Users/user/Downloads/126.eye/01-1.data/Training/01.data/TS/003/T1/image_30_face_crop",
]
ZEROSHOT_CSV = r"C:\Users\user\Desktop\brainbuddy_AI\labels_zeroshot.csv"
EMB_PATH = "embeddings.npz"

# ====== 임베딩/모델 설정 ======
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".webp")
MAX_FRAMES = 30
BATCH_SIZE = 32

# ====== 학습 파라미터 ======
Q_EXTREME = 0.20      # 상/하위 20%씩 pos/neg로 사용 (0.10~0.30 사이 조절 가능)
VAL_RATIO = 0.2
TARGET_RECALL = 0.90
RANDOM_SEED = 42

# ===================== 유틸 함수들 =====================

def list_seq_folders(roots):
    seqs = []
    for rd in roots:
        rd = os.path.abspath(rd)
        if not os.path.exists(rd): continue
        for cur, _, files in os.walk(rd):
            if any(f.lower().endswith(IMG_EXTS) for f in files):
                seqs.append(cur)
    return sorted(set(seqs))

def list_images(folder):
    try:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
        return [os.path.join(folder, f) for f in files]
    except:
        return []

def even_sample(items, k):
    if k<=0 or len(items)<=k: return items
    idx = np.linspace(0, len(items)-1, k, dtype=int)
    return [items[i] for i in idx]

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
    )
    model.eval()
    return model, preprocess

@torch.no_grad()
def embed_folder(model, preprocess, folder):
    imgs = list_images(folder)
    if not imgs: return None
    sel = even_sample(imgs, MAX_FRAMES)
    xs = []
    for p in sel:
        try:
            xs.append(preprocess(Image.open(p).convert("RGB")))
        except:
            pass
    if not xs: return None
    X = torch.stack(xs,0).to(DEVICE)
    embs = []
    for i in range(0, len(X), BATCH_SIZE):
        v = model.encode_image(X[i:i+BATCH_SIZE])
        v = v / v.norm(dim=-1, keepdim=True)
        embs.append(v)
    v = torch.cat(embs,0).mean(0).cpu().numpy()
    return v

def build_embeddings(folders, save_npz=EMB_PATH):
    print("[Build] embeddings...")
    model, preprocess = load_model()
    X, ids = [], []
    for fd in tqdm(folders, desc="Embedding folders"):
        v = embed_folder(model, preprocess, fd)
        if v is None: continue
        X.append(v); ids.append(fd)
    X = np.stack(X,0)
    np.savez(save_npz, X=X, ids=np.array(ids, dtype=object),
             MODEL_NAME=MODEL_NAME, PRETRAINED=PRETRAINED)
    print(f"[Saved] {save_npz}  (N={len(ids)}, D={X.shape[1]})")
    return X, ids

def load_embeddings(npz_path=EMB_PATH):
    if not os.path.exists(npz_path):
        folders = list_seq_folders(ROOT_DIRS)
        return build_embeddings(folders, npz_path)
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]; ids = data["ids"]
    print(f"[Load] embeddings: N={len(ids)}, D={X.shape[1]}")
    return X, ids

# 그룹 추출: '.../<group>/<sequence_folder>' 구조에서 group을 상위 폴더명으로
def extract_group(folder_path: str) -> str:
    p = os.path.normpath(folder_path)
    parts = p.split(os.sep)
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]

def split_train_val_grouped(X, y, folders, val_ratio=0.2, seed=42):
    groups = np.array([extract_group(f) for f in folders])
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(gss.split(X, y, groups))
    return (X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            [folders[i] for i in tr_idx], [folders[i] for i in va_idx])

# ===================== 라벨 준비 =====================

def prepare_labels_extreme(df, q=Q_EXTREME):
    """
    극단 분위수로 pos/neg 선택 (부호 무시):
    - margin 상위 (1-q) 분위수 이상 → pos(1)
    - margin 하위 q 분위수 이하  → neg(0)
    """
    need = ["folder","margin","p_focused"]
    df = df.dropna(subset=need).copy()

    low_thr  = float(np.quantile(df["margin"], q))
    high_thr = float(np.quantile(df["margin"], 1.0 - q))
    print(f"[Quantiles-extreme] low_thr={low_thr:.6f}, high_thr={high_thr:.6f}")

    neg_df = df[df["margin"] <= low_thr].copy()
    pos_df = df[df["margin"] >= high_thr].copy()

    neg_df["y"] = 0
    pos_df["y"] = 1

    # 균형 맞추기
    n = min(len(pos_df), len(neg_df))
    pos_df = pos_df.sample(n=n, random_state=RANDOM_SEED)
    neg_df = neg_df.sample(n=n, random_state=RANDOM_SEED)

    both = pd.concat([pos_df, neg_df], ignore_index=True).drop_duplicates(subset=["folder"])
    print(f"[Class sizes (extreme)] pos={len(pos_df)}, neg={len(neg_df)}")
    return both

def align_Xy(X, ids, df_lab):
    id2idx = {ids[i]: i for i in range(len(ids))}
    keep = [r.folder in id2idx for r in df_lab.itertuples()]
    df_lab = df_lab[keep].copy()
    idx = [id2idx[r.folder] for r in df_lab.itertuples()]
    X_sel = X[idx]
    y_sel = df_lab["y"].astype(int).values
    folders_sel = df_lab["folder"].tolist()
    return X_sel, y_sel, folders_sel

# ===================== 튜닝/학습 =====================

def tune_threshold_for_recall(y_true, p_prob, target_recall=0.90):
    grid = np.linspace(0.1, 0.9, 161)
    best = None
    for th in grid:
        pred = (p_prob >= th).astype(int)
        tp = ((pred==1)&(y_true==1)).sum()
        fp = ((pred==1)&(y_true==0)).sum()
        fn = ((pred==0)&(y_true==1)).sum()
        recall = tp / max(tp+fn,1)
        precision = tp / max(tp+fp,1)
        score = (recall >= target_recall, precision, recall)
        if best is None or score > best[0]:
            best = (score, th, recall, precision)
    _, th, r, p = best
    return float(th), float(r), float(p)

def main():
    # (0) 임베딩 로드/생성
    X, ids = load_embeddings(EMB_PATH)

    # (1) 라벨 준비 (극단 분위수)
    if not os.path.exists(ZEROSHOT_CSV):
        raise FileNotFoundError(f"CSV not found: {ZEROSHOT_CSV}")
    df = pd.read_csv(ZEROSHOT_CSV)

    df_lab = prepare_labels_extreme(df, q=Q_EXTREME)
    X_all, y_all, folders_all = align_Xy(X, ids, df_lab)
    print(f"[Dataset aligned] N={len(y_all)}, pos={(y_all==1).sum()}, neg={(y_all==0).sum()}")

    # (2) 그룹 기반 학습/검증 분할 (누수 방지)
    Xtr, ytr, Xva, yva, ftr, fva = split_train_val_grouped(
        X_all, y_all, folders_all, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )
    print("[Groups] train unique groups:", len(set(map(extract_group, ftr))),
          "val unique groups:", len(set(map(extract_group, fva))))

    # (3) 로지스틱 학습
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    # (4) 검증 성능
    pred = clf.predict(Xva)
    prob = clf.predict_proba(Xva)[:,1]
    print("=== Validation Report ===")
    print(classification_report(yva, pred, digits=4))
    try:
        auc = roc_auc_score(yva, prob)
        print("ROC-AUC:", round(auc,4))
    except:
        pass

    th, r, p = tune_threshold_for_recall(yva, prob, target_recall=TARGET_RECALL)
    print(f"[Tuned] threshold={th:.3f}  (recall≈{r:.3f}, precision≈{p:.3f})")

    # (5) 저장
    np.savez("linear_probe.npz",
             W=clf.coef_, b=clf.intercept_, classes_=clf.classes_,
             MODEL_NAME=MODEL_NAME, PRETRAINED=PRETRAINED)
    with open("probe_meta.json","w",encoding="utf-8") as f:
        json.dump({
            "threshold": float(th),
            "embedding_dim": int(X.shape[1]),
            "train_size": int(len(ytr)),
            "val_size": int(len(yva)),
            "pos_count": int((y_all==1).sum()),
            "neg_count": int((y_all==0).sum()),
            "model": MODEL_NAME, "pretrained": PRETRAINED
        }, f, ensure_ascii=False, indent=2)
    print("[Saved] linear_probe.npz, probe_meta.json")

if __name__ == "__main__":
    main()
