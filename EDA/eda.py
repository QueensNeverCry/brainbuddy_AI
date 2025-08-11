"""
Focus EDA & Visualization — Patched Full Script v2
--------------------------------------------------
Fixes included:
- SSIM crash fix: all frames resized to a common size in the loader + extra guards in SSIM code
- Matplotlib 3.9+ deprecation: boxplot `labels` → `tick_labels`
- Fusion 5D (sequence-level) features supported: [yaw_diff, pitch_diff, yawn_detected, eye_closed_ratio, head_speed]
- Defensive checks for empty inputs
- Optional CNN feature extraction → t‑SNE/UMAP (if PyTorch available)

HOW TO USE:
1) Set your pickle lists in main(): each pickle contains a list of (folder_path, label)
2) Ensure each folder has ≥30 images (jpg/jpeg/png)
3) (Optional) Put `fusion_features.pkl` (length 5 array) in the folder to enable sequence-level feature plots
4) Run: python eda.py
Figures saved to ./eda_out by default
"""

import os
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from skimage.metrics import structural_similarity as ssim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional: UMAP and PyTorch
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torchvision.transforms as T  # type: ignore
    import torchvision.models as models  # type: ignore
    from PIL import Image  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


# ===============================
# -------- Data Structure -------
# ===============================
@dataclass
class SequenceData:
    """Container for a 30‑frame sequence and auxiliary info used by EDA.
    frames: list of raw frames (HxWxC, BGR — cv2)
    face_bboxes: per‑frame face rectangles (optional)
    ear/yawn_prob/head_movement: per‑frame arrays (optional, may be NaN)
    fusion_seq: sequence‑level 5‑D vector [yaw_diff, pitch_diff, yawn_detected,
               eye_closed_ratio, head_speed] (optional)
    """
    seq_id: str
    label: int
    frames: List[np.ndarray]
    face_bboxes: List[Tuple[int, int, int, int]]
    ear: List[float]
    yawn_prob: List[float]
    head_movement: List[float]
    fusion_seq: Optional[List[float]] = None


# ===============================
# ----------- I/O Utils ---------
# ===============================

def load_data(pkl_files: List[str]) -> List[Tuple[str, int]]:
    """Load (folder_path, label) tuples from multiple pickle files."""
    all_data: List[Tuple[str, int]] = []
    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            all_data.extend(data)
    return all_data


def _load_optional_pickle(path: str):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as fp:
                return pickle.load(fp)
        except Exception:
            return None
    return None


def _read_frames_from_folder(folder_path: str, max_frames: int = 30,
                             resize_to: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """Read first max_frames from folder and resize to a common size (for SSIM safety)."""
    img_files = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:max_frames]
    frames: List[np.ndarray] = []
    for f in img_files:
        p = os.path.join(folder_path, f)
        img = cv2.imread(p)  # BGR
        if img is None:
            continue
        if resize_to is not None:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        frames.append(img)
    return frames


# ===============================
# ------- Feature Helpers -------
# ===============================

def compute_blur_score(frame: np.ndarray) -> float:
    """Variance of Laplacian (higher is sharper)."""
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(frame: np.ndarray) -> float:
    """Mean value of V channel in HSV space (0–255)."""
    if frame.ndim == 3:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return float(hsv[..., 2].mean())
    return float(frame.mean())


def frame_bbox_stats(bbox):
    # bbox = (x,y,w,h) or None
    if bbox and bbox[2] > 0 and bbox[3] > 0:
        return bbox
    return (0, 0, 0, 0)


# ===============================
# -------- Plotting Utils -------
# ===============================

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def save_fig(path):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hist(data, title, xlabel, out_path, bins=30):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_fig(out_path)


def plot_box(data_groups: Dict[str, List[float]], title, ylabel, out_path):
    plt.figure()
    labels = list(data_groups.keys())
    data = [data_groups[k] for k in labels]
    # Matplotlib 3.9+: use tick_labels
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    save_fig(out_path)


def plot_line(series: List[float], title, xlabel, ylabel, out_path):
    plt.figure()
    plt.plot(series)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(out_path)


def plot_quiver(flow: np.ndarray, step: int, title: str, out_path: str):
    # flow shape: HxWx2 (u,v)
    H, W = flow.shape[:2]
    y, x = np.mgrid[0:H:step, 0:W:step]
    u = flow[::step, ::step, 0]
    v = flow[::step, ::step, 1]
    plt.figure()
    plt.quiver(x, y, u, v)
    plt.gca().invert_yaxis()
    plt.title(title)
    save_fig(out_path)


def plot_heatmap(mat: np.ndarray, title: str, out_path: str, xlabel: str = "", ylabel: str = ""):
    plt.figure()
    plt.imshow(mat, aspect='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    save_fig(out_path)


def grouped_bar(means: Dict[str, List[float]], labels: List[str], title: str, out_path: str):
    fig = plt.figure()
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(means))
    for i, (feat, vals) in enumerate(means.items()):
        plt.bar(x + i * width, vals, width)
    plt.xticks(x + (len(means) - 1) * width / 2, labels)
    plt.title(title)
    plt.ylabel("Mean")
    save_fig(out_path)


# ===============================
# ------------- Loaders ---------
# ===============================

def load_sequences_and_labels_from_folders(data_list: List[Tuple[str, int]], frames_per_seq: int = 30,
                                           resize_to: Tuple[int, int] = (224, 224), verbose: bool = False) -> List[SequenceData]:
    """Load sequences given a list of (folder_path, label).
    - Read first `frames_per_seq` images per folder
    - Resize all frames to `resize_to` to avoid SSIM shape errors
    - Load optional face_bboxes.pkl and fusion_features.pkl
    - fusion_features.pkl (sequence‑level 5D) → `fusion_seq`
    - If per‑frame dict exists (ear/yawn_prob/head_movement), it's used; else NaN lists
    """
    seqs: List[SequenceData] = []
    for folder_path, label in tqdm(data_list, desc="Load sequences"):
        if not os.path.isdir(folder_path):
            if verbose: print(f"[SKIP] no dir: {folder_path}")
            continue
        frames = _read_frames_from_folder(folder_path, frames_per_seq, resize_to=resize_to)
        if len(frames) < frames_per_seq:
            if verbose: print(f"[SKIP] too few frames({len(frames)}): {folder_path}")
            continue

        # face bboxes (optional)
        bboxes = _load_optional_pickle(os.path.join(folder_path, "face_bboxes.pkl"))
        if not isinstance(bboxes, list) or len(bboxes) < frames_per_seq:
            bboxes = [(0, 0, 0, 0)] * frames_per_seq

        # behavior features
        fusion = _load_optional_pickle(os.path.join(folder_path, "fusion_features.pkl"))
        ear = [np.nan] * frames_per_seq
        yawn = [np.nan] * frames_per_seq
        head = [np.nan] * frames_per_seq
        fusion_seq: Optional[List[float]] = None

        if isinstance(fusion, dict):
            if isinstance(fusion.get('ear'), (list, np.ndarray)):
                ear = list(fusion['ear'])[:frames_per_seq] + [np.nan] * (frames_per_seq - len(fusion['ear']))
            if isinstance(fusion.get('yawn_prob'), (list, np.ndarray)):
                yawn = list(fusion['yawn_prob'])[:frames_per_seq] + [np.nan] * (frames_per_seq - len(fusion['yawn_prob']))
            if isinstance(fusion.get('head_movement'), (list, np.ndarray)):
                head = list(fusion['head_movement'])[:frames_per_seq] + [np.nan] * (frames_per_seq - len(fusion['head_movement']))
        elif isinstance(fusion, (list, np.ndarray)) and len(fusion) == 5:
            # Your format: [yaw_diff, pitch_diff, yawn_detected, eye_closed_ratio, head_speed]
            fusion_seq = [float(x) for x in fusion]
            # If you want per‑frame plots visible even without per‑frame signals, you could replicate:
            # ear  = [fusion_seq[3]] * frames_per_seq          # eye_closed_ratio
            # yawn = [fusion_seq[2]] * frames_per_seq          # yawn_detected (0/1)
            # head = [fusion_seq[4]] * frames_per_seq          # head_speed

        seqs.append(SequenceData(
            seq_id=os.path.basename(folder_path.rstrip(os.sep)),
            label=int(label),
            frames=frames,
            face_bboxes=bboxes[:frames_per_seq],
            ear=ear,
            yawn_prob=yawn,
            head_movement=head,
            fusion_seq=fusion_seq,
        ))
    if verbose: print(f"[INFO] Loaded sequences: {len(seqs)}")
    return seqs


# ===============================
# -------------- EDA -------------
# ===============================

def eda_frame_quality(seqs: List[SequenceData], out_dir: str):
    """(2) Frame quality: blur, brightness, face size ratio."""
    if not seqs:
        print("[WARN] eda_frame_quality: no sequences, skipping.")
        return
    blur_scores, brightness, face_area_ratio = [], [], []

    # Assume constant canvas size after loader resize
    H, W = seqs[0].frames[0].shape[:2]

    for s in tqdm(seqs, desc="Frame quality"):
        for f, bbox in zip(s.frames, s.face_bboxes):
            blur_scores.append(compute_blur_score(f))
            brightness.append(compute_brightness(f))
            x, y, w, h = frame_bbox_stats(bbox)
            area = (w * h) / (W * H) if W * H > 0 else 0.0
            face_area_ratio.append(area)

    plot_hist(blur_scores, "Blur (Variance of Laplacian)", "Blur score",
              os.path.join(out_dir, "2_quality_blur_hist.png"))
    plot_hist(brightness, "Brightness (HSV V-mean)", "Brightness",
              os.path.join(out_dir, "2_quality_brightness_hist.png"))
    plot_hist(face_area_ratio, "Face area ratio", "Area ratio",
              os.path.join(out_dir, "2_quality_face_area_hist.png"))


def eda_behavior_stats(seqs: List[SequenceData], out_dir: str, label_names: Dict[int, str]):
    """(3) Behavior/Expression stats.
    - Per‑frame channels (if provided): head_movement, EAR, yawn_prob → boxplots by label
    - Sequence‑level fusion (5D): yaw_diff, pitch_diff, yawn_detected, eye_closed_ratio, head_speed
      → boxplots (continuous) + rate bars (binary yawn_detected)
    """
    if not seqs:
        print("[WARN] eda_behavior_stats: no sequences, skipping.")
        return

    by_label_pf: Dict[int, Dict[str, List[float]]] = {}
    by_label_seq: Dict[int, Dict[str, List[float]]] = {}

    for s in seqs:
        by_label_pf.setdefault(s.label, {"head": [], "ear": [], "yawn": []})
        by_label_pf[s.label]["head"].extend([v for v in s.head_movement if not np.isnan(v)])
        by_label_pf[s.label]["ear"].extend([v for v in s.ear if not np.isnan(v)])
        by_label_pf[s.label]["yawn"].extend([v for v in s.yawn_prob if not np.isnan(v)])

        if s.fusion_seq is not None and len(s.fusion_seq) == 5:
            d = by_label_seq.setdefault(s.label, {
                "yaw_diff": [], "pitch_diff": [], "yawn_detected": [],
                "eye_closed_ratio": [], "head_speed": []
            })
            d["yaw_diff"].append(s.fusion_seq[0])
            d["pitch_diff"].append(s.fusion_seq[1])
            d["yawn_detected"].append(s.fusion_seq[2])
            d["eye_closed_ratio"].append(s.fusion_seq[3])
            d["head_speed"].append(s.fusion_seq[4])

    # Per‑frame boxplots (only if data exists)
    if any(len(v["head"]) for v in by_label_pf.values()):
        data = {label_names.get(k, str(k)): v["head"] for k, v in by_label_pf.items()}
        plot_box(data, "Head movement distribution by label", "movement",
                 os.path.join(out_dir, "3_behavior_head_box.png"))
    if any(len(v["ear"]) for v in by_label_pf.values()):
        data = {label_names.get(k, str(k)): v["ear"] for k, v in by_label_pf.items()}
        plot_box(data, "EAR distribution by label", "EAR",
                 os.path.join(out_dir, "3_behavior_ear_box.png"))
    if any(len(v["yawn"]) for v in by_label_pf.values()):
        data = {label_names.get(k, str(k)): v["yawn"] for k, v in by_label_pf.items()}
        plot_box(data, "Yawn prob distribution by label", "yawn prob",
                 os.path.join(out_dir, "3_behavior_yawn_box.png"))

    # Sequence‑level fusion
    if by_label_seq:
        for k in ["yaw_diff", "pitch_diff", "eye_closed_ratio", "head_speed"]:
            groups = {label_names.get(l, str(l)): by_label_seq[l][k] for l in by_label_seq}
            if any(len(v) > 0 for v in groups.values()):
                plot_box(groups, f"{k} distribution by label", k,
                         os.path.join(out_dir, f"3_behavior_{k}_box.png"))
        labels_sorted = sorted(by_label_seq.keys())
        means = {"yawn_detected": [float(np.mean(by_label_seq[l]["yawn_detected"]))
                                     if len(by_label_seq[l]["yawn_detected"]) > 0 else float("nan")
                                     for l in labels_sorted]}
        disp = [label_names.get(l, str(l)) for l in labels_sorted]
        grouped_bar(means, disp, "Yawn detected rate by label",
                    os.path.join(out_dir, "3_behavior_yawn_detected_rate.png"))


def eda_sequence_dynamics(seqs: List[SequenceData], out_dir: str):
    """(4) Sequence dynamics: SSIM curve/matrix + optical flow sample."""
    if not seqs:
        print("[WARN] eda_sequence_dynamics: no sequences, skipping.")
        return

    # pick one per label for visualization brevity
    chosen: Dict[int, SequenceData] = {}
    for s in seqs:
        if s.label not in chosen:
            chosen[s.label] = s

    for lbl, s in chosen.items():
        # SSIM vs first frame — dtype/shape guard + data_range
        gray0 = cv2.cvtColor(s.frames[0], cv2.COLOR_BGR2GRAY) if s.frames[0].ndim == 3 else s.frames[0]
        if gray0.dtype != np.uint8:
            gray0 = gray0.astype(np.uint8)
        target_h, target_w = gray0.shape[:2]

        ssim_curve: List[float] = []
        for f in s.frames:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            if g.dtype != np.uint8:
                g = g.astype(np.uint8)
            if g.shape[:2] != (target_h, target_w):
                g = cv2.resize(g, (target_w, target_h), interpolation=cv2.INTER_AREA)
            score = ssim(gray0, g, data_range=255)
            ssim_curve.append(score)
        if ssim_curve:
            plot_line(ssim_curve, f"SSIM vs first frame (label {lbl})", "frame idx", "SSIM",
                      os.path.join(out_dir, f"4_dyn_ssim_curve_label{lbl}.png"))
        else:
            print(f"[WARN] Empty SSIM curve for seq {s.seq_id} (label {lbl}).")

        # SSIM matrix (subsample to <=30 frames for speed)
        step = max(1, len(s.frames) // 30)
        frames_ds = s.frames[::step]
        N = len(frames_ds)
        if N >= 2:
            grays: List[np.ndarray] = []
            for fr in frames_ds:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr
                if g.dtype != np.uint8:
                    g = g.astype(np.uint8)
                if g.shape[:2] != (target_h, target_w):
                    g = cv2.resize(g, (target_w, target_h), interpolation=cv2.INTER_AREA)
                grays.append(g)
            mat = np.zeros((N, N), dtype=np.float32)
            for i in range(N):
                for j in range(N):
                    mat[i, j] = ssim(grays[i], grays[j], data_range=255)
            plot_heatmap(mat, f"SSIM matrix (label {lbl})",
                         os.path.join(out_dir, f"4_dyn_ssim_mat_label{lbl}.png"), "frame", "frame")
        else:
            print(f"[WARN] SSIM matrix skipped for label {lbl}: too few frames (N={N})")

        # Optical Flow (one step around mid index) — requires >=2 frames
        if len(s.frames) >= 2:
            mid = len(s.frames) // 2
            g1 = cv2.cvtColor(s.frames[mid - 1], cv2.COLOR_BGR2GRAY) if s.frames[mid - 1].ndim == 3 else s.frames[mid - 1]
            g2 = cv2.cvtColor(s.frames[mid], cv2.COLOR_BGR2GRAY) if s.frames[mid].ndim == 3 else s.frames[mid]
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            plot_quiver(flow, step=10, title=f"Optical Flow quiver (label {lbl})",
                        out_path=os.path.join(out_dir, f"4_dyn_flow_quiver_label{lbl}.png"))

    # Example per‑sequence motion curve (head_movement) if available
    seq0 = seqs[0]
    if any(not np.isnan(v) for v in seq0.head_movement):
        plot_line(seq0.head_movement, "Head movement over time (example seq)",
                  "frame idx", "movement",
                  os.path.join(out_dir, "4_dyn_head_movement_line.png"))


def eda_label_feature_relations(seqs: List[SequenceData], out_dir: str, label_names: Dict[int, str]):
    """(5) Label–feature relations: correlation heatmap + grouped mean bars."""
    if not seqs:
        print("[WARN] eda_label_feature_relations: no sequences, skipping.")
        return

    rows = []
    for s in seqs:
        row = {
            "seq_id": s.seq_id,
            "label": s.label,
            # per‑frame aggregates (may be NaN‑only)
            "head_mean": safe_nanmean(s.head_movement),
            "ear_mean":  safe_nanmean(s.ear),
            "yawn_mean": safe_nanmean(s.yawn_prob),
            # sequence‑level fusion (if present)
            "yaw_diff": np.nan, "pitch_diff": np.nan,
            "yawn_detected": np.nan, "eye_closed_ratio": np.nan, "head_speed": np.nan,
        }
        if s.fusion_seq is not None and len(s.fusion_seq) == 5:
            row.update({
                "yaw_diff": float(s.fusion_seq[0]),
                "pitch_diff": float(s.fusion_seq[1]),
                "yawn_detected": float(s.fusion_seq[2]),
                "eye_closed_ratio": float(s.fusion_seq[3]),
                "head_speed": float(s.fusion_seq[4]),
            })
        rows.append(row)
    df = pd.DataFrame(rows)

    if df.empty or "label" not in df.columns:
        print("[WARN] Empty DataFrame or missing label column, skipping relations plots.")
        return

    cols = [c for c in [
        "head_mean", "ear_mean", "yawn_mean",
        "yaw_diff", "pitch_diff", "eye_closed_ratio", "yawn_detected", "head_speed",
        "label"
    ] if c in df.columns]

    corr = df[cols].corr(numeric_only=True).values
    plot_heatmap(corr, "Correlation heatmap (features vs label)",
                 os.path.join(out_dir, "5_rel_corr_heatmap.png"), "vars", "vars")

    labels_sorted = sorted(df["label"].unique().tolist())
    disp_labels = [label_names.get(l, str(l)) for l in labels_sorted]
    features_for_bar = [
        "head_mean", "ear_mean", "yawn_mean",
        "yaw_diff", "pitch_diff", "eye_closed_ratio", "head_speed", "yawn_detected"
    ]

    means = {feat: [] for feat in features_for_bar}
    for l in labels_sorted:
        sub = df[df.label == l]
        for feat in features_for_bar:
            if feat in df.columns:
                means[feat].append(safe_nanmean(sub[feat]) if len(sub) > 0 else float("nan"))

    grouped_bar(means, disp_labels, "Feature means per label",
                os.path.join(out_dir, "5_rel_grouped_bar.png"))


# ===============================
# --- CNN Feature Extraction ----
# ===============================

# --- CNN Feature Extraction (MobileNet_v2) ---
def cnn_features_from_folders(
    data_list: List[Tuple[str, int]],
    frames_per_seq: int = 30,
    batch_size: int = 32,
    device: Optional[str] = None
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Compute frame-level CNN features using torchvision MobileNet_v2
    (global-average pooled 1280-D vectors).

    Returns
        features_frames: (N_frames, 1280)
        frame_seq_ids   : list of sequence id per frame
        frame_labels    : list of labels per frame
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch/torchvision not available. Install to use CNN feature extraction.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ---- MobileNet_v2 backbone ----
    mb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Feature extractor that outputs the 1280-D pooled features
    feature_extractor = torch.nn.Sequential(
        mb.features,                    # conv stack
        torch.nn.AdaptiveAvgPool2d(1),  # global avg pool to (C,1,1)
        torch.nn.Flatten()              # -> (C,)
    ).eval().to(device)
    feat_dim = 1280

    all_feats: List[np.ndarray] = []
    all_seq_ids: List[str] = []
    all_labels: List[int] = []
    batch_imgs: List[torch.Tensor] = []
    batch_meta: List[Tuple[str, int]] = []  # (seq_id, label)

    for folder_path, label in tqdm(data_list, desc="CNN features (MobileNet_v2)"):
        if not os.path.isdir(folder_path):
            continue
        seq_id = os.path.basename(folder_path.rstrip(os.sep))
        img_files = sorted([f for f in os.listdir(folder_path)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:frames_per_seq]
        for f in img_files:
            p = os.path.join(folder_path, f)
            bgr = cv2.imread(p)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = tfm(pil)
            batch_imgs.append(tensor)
            batch_meta.append((seq_id, int(label)))

            if len(batch_imgs) == batch_size:
                inp = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    feats = feature_extractor(inp).cpu().numpy()
                all_feats.append(feats)
                for (sid, lab) in batch_meta:
                    all_seq_ids.append(sid)
                    all_labels.append(lab)
                batch_imgs, batch_meta = [], []

    if batch_imgs:
        inp = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = feature_extractor(inp).cpu().numpy()
        all_feats.append(feats)
        for (sid, lab) in batch_meta:
            all_seq_ids.append(sid)
            all_labels.append(lab)

    features_frames = np.vstack(all_feats) if len(all_feats) > 0 else np.zeros((0, feat_dim))
    return features_frames, all_seq_ids, all_labels



def visualize_sequence_embeddings_from_features(features_frames: np.ndarray,
                                                frame_seq_ids: List[str],
                                                frame_labels: List[int],
                                                out_dir: str):
    """Aggregate frame‑level features to sequence‑level and visualize with t‑SNE/UMAP."""
    ensure_dir(out_dir)
    df_meta = pd.DataFrame({"seq_id": frame_seq_ids, "label": frame_labels})

    # Aggregate by mean over frames per sequence
    df_feats = pd.DataFrame(features_frames)
    df_feats["seq_id"] = frame_seq_ids
    agg = df_feats.groupby("seq_id").mean()
    seq_label = df_meta.groupby("seq_id")["label"].agg(lambda x: int(pd.Series(x).mode().iloc[0]))

    X = agg.values
    y = seq_label.loc[agg.index].values

    # Standardize + PCA(<=50) for stability
    Xs = StandardScaler().fit_transform(X)
    pca_dim = min(50, Xs.shape[1])
    Xp = PCA(n_components=pca_dim).fit_transform(Xs)

    # t‑SNE 2D
    # t-SNE 2D (version-compatible)
    try:
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, len(Xp)//10)),
            n_iter=1000,  # some sklearns accept this here
        )
    except TypeError:
        # Older sklearn: no n_iter in __init__
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, len(Xp)//10)),
        )
    Zt = tsne.fit_transform(Xp)


    plt.figure()
    for l in np.unique(y):
        idx = np.where(y == l)[0]
        plt.scatter(Zt[idx, 0], Zt[idx, 1], label=str(l))
    plt.legend()
    plt.title("t-SNE (sequence-level CNN features)")
    save_fig(os.path.join(out_dir, "tsne_seq_features.png"))

    # UMAP 2D (if available)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        Zu = reducer.fit_transform(Xp)
        plt.figure()
        for l in np.unique(y):
            idx = np.where(y == l)[0]
            plt.scatter(Zu[idx, 0], Zu[idx, 1], label=str(l))
        plt.legend()
        plt.title("UMAP (sequence-level CNN features)")
        save_fig(os.path.join(out_dir, "umap_seq_features.png"))

def safe_nanmean(x) -> float:
    try:
        arr = np.asarray(list(x), dtype=float)
    except Exception:
        return float("nan")
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


# ===============================
# -------------- main ------------
# ===============================
if __name__ == "__main__":
    # ---- 0) Configure your data paths here ----
    train_pkl_files = [
        r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_01.pkl",
        r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/train/20_03.pkl",
    ]
    val_pkl_files = [
        r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_01.pkl",
        r"C:/KSEB/brainbuddy_AI/preprocess2/pickle_labels/valid/20_03.pkl",
    ]

    frames_per_seq = 30
    out_dir = "./eda_out"
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) Load (folder_path, label) lists from pickles ----
    train_data_list = load_data(train_pkl_files) if len(train_pkl_files) > 0 else []
    val_data_list   = load_data(val_pkl_files) if len(val_pkl_files) > 0 else []

    # You can start with a subset to test quickly
    data_list = train_data_list[:200] if len(train_data_list) > 200 else train_data_list
    label_names = {0: "low", 1: "high"}  # edit to your classes

    # ---- 2) Build SequenceData objects ----
    seqs = load_sequences_and_labels_from_folders(data_list, frames_per_seq=frames_per_seq, resize_to=(224,224))
    if len(seqs) == 0:
        print("[ERROR] No sequences loaded. Check pkl contents, folder paths, and that each folder has >=30 images.")
        raise SystemExit(1)

    # ---- 3) EDA: (2) Frame quality ----
    eda_frame_quality(seqs, out_dir)

    # ---- 4) EDA: (3) Behavior/Expression stats ----
    eda_behavior_stats(seqs, out_dir, label_names)

    # ---- 5) EDA: (4) Sequence dynamics ----
    eda_sequence_dynamics(seqs, out_dir)

    # ---- 6) EDA: (5) Label–feature relations ----
    eda_label_feature_relations(seqs, out_dir, label_names)

    # ---- 7) (Optional) CNN features → t‑SNE/UMAP ----
    if HAS_TORCH and len(data_list) > 0:
        feats, frame_seq_ids, frame_labels = cnn_features_from_folders(data_list, frames_per_seq=frames_per_seq, batch_size=64)
        visualize_sequence_embeddings_from_features(feats, frame_seq_ids, frame_labels, out_dir)
    else:
        print("[Info] Skipping CNN features/t-SNE because PyTorch is unavailable or data_list is empty.")

    print(f"Done. Figures saved under: {os.path.abspath(out_dir)}")
