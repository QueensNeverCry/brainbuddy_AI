# make_eval_figs.py
# Generates 3 images: confusion_matrix.png, roc_curve.png, metrics_card.png
# - Option A: run model on test split (labels_final.csv + splits_indices.json + the_best.pth)
# - Option B: skip model and load precomputed predictions CSV with columns: y_true,y_prob

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

# ===== Your model class must live here =====
# Ensure clip2_1.py defines CNN_LSTM that takes (B,T,C,H,W) and returns (B,) or (B,1)
from clip2_1 import CNN_LSTM


# -----------------------------
# Utilities
# -----------------------------
def build_eval_tfms(img_size=224):
    return transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def load_image_seq(folder, tfm, seq_len=30):
    files = sorted([f for f in os.listdir(folder)
                    if f.lower().endswith((".png",".jpg",".jpeg"))])
    if len(files) == 0:
        raise FileNotFoundError(f"No images in {folder}")
    # repeat_last padding
    if len(files) < seq_len:
        files = files + [files[-1]] * (seq_len - len(files))
    else:
        files = files[:seq_len]
    frames = []
    for fn in files:
        img = Image.open(os.path.join(folder, fn)).convert("RGB")
        frames.append(tfm(img))
    return torch.stack(frames, dim=0)  # (T,C,H,W)

def pick_test_df(df, split_json_path):
    """Pick test rows flexibly based on splits_indices.json content."""
    if not split_json_path or not Path(split_json_path).exists():
        print("[WARN] split_json missing -> using ALL rows for evaluation.")
        return df.reset_index(drop=True)
    with open(split_json_path, "r", encoding="utf-8") as f:
        sp = json.load(f)

    if "test_indices" in sp and isinstance(sp["test_indices"], list):
        idx = sp["test_indices"]; print(f"[INFO] Using test_indices: n={len(idx)}")
        return df.iloc[idx].reset_index(drop=True)

    if "test" in sp and isinstance(sp["test"], list) and all(isinstance(i,int) for i in sp["test"]):
        idx = sp["test"]; print(f"[INFO] Using splits['test']: n={len(idx)}")
        return df.iloc[idx].reset_index(drop=True)

    if "y_counts" in sp and "test" in sp["y_counts"]:
        n = sp["y_counts"]["test"].get("n", None)
        print(f"[WARN] Only test counts present (n={n}). Using ALL rows.")
        return df.reset_index(drop=True)

    print("[WARN] Unknown split_json structure. Using ALL rows.")
    return df.reset_index(drop=True)


# -----------------------------
# Core evaluation
# -----------------------------
def run_model_and_collect(args):
    """Run model on test split and return y_true, y_prob arrays."""
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    tfm = build_eval_tfms(args.img_size)

    df_all = pd.read_csv(args.labels_csv)
    df_test = pick_test_df(df_all, args.split_json)

    # Load model
    model = CNN_LSTM()
    ckpt = torch.load(args.ckpt, map_location=device)
    # try to be flexible with checkpoint formats
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.to(device)
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for _, row in df_test.iterrows():
            folder = row["folder"]
            label = int(row["predicted_label"])
            x = load_image_seq(folder, tfm, seq_len=args.seq_len).unsqueeze(0).to(device)  # (1,T,C,H,W)
            logits = model(x)
            if logits.ndim == 2:
                logits = logits.squeeze(1)
            prob = torch.sigmoid(logits).item()
            y_true.append(label)
            y_prob.append(prob)
    return np.array(y_true), np.array(y_prob)


def load_preds_from_csv(pred_csv):
    """Read y_true,y_prob from a CSV (columns must be named y_true,y_prob)."""
    df = pd.read_csv(pred_csv)
    if not {"y_true","y_prob"} <= set(df.columns):
        raise ValueError("pred_csv must contain columns: y_true,y_prob")
    return df["y_true"].to_numpy().astype(int), df["y_prob"].to_numpy().astype(float)


# -----------------------------
# Plotters
# -----------------------------
def save_confusion_matrix(y_true, y_pred, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = plt.figure(figsize=(5.5, 4.8))
    ax = plt.gca()
    im = ax.imshow(cm)  # no explicit colormap to stick to style requirements

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center")

    ax.set_xlabel("Predicted Label (0=Unfocused, 1=Focused)")
    ax.set_ylabel("True Label (0=Unfocused, 1=Focused)")
    ax.set_title("Confusion Matrix (Test Set)")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def save_roc_curve(y_true, y_prob, out_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    fig = plt.figure(figsize=(5.5, 4.8))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def save_metrics_card(y_true, y_pred, y_prob, out_path="metrics_card.png", threshold=0.5):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0,1], zero_division=0
    )
    f1_macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    # Make a clean metric "card" as a single image
    fig = plt.figure(figsize=(6.3, 4.6))
    plt.axis("off")
    lines = [
        "Test Metrics (Threshold = {:.2f})".format(threshold),
        "",
        f"Accuracy : {acc:.4f}",
        f"F1 (macro): {f1_macro:.4f}",
        f"AUC      : {auc:.4f}",
        "",
        "Class-wise (0 = Unfocused, 1 = Focused)",
        f"Class 0  - Precision: {prec[0]:.4f} | Recall: {rec[0]:.4f} | F1: {f1[0]:.4f}",
        f"Class 1  - Precision: {prec[1]:.4f} | Recall: {rec[1]:.4f} | F1: {f1[1]:.4f}",
    ]
    plt.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--labels_csv", default="labels_final.csv")
    ap.add_argument("--split_json", default="splits_indices.json")
    ap.add_argument("--ckpt", default="the_best.pth")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--threshold", type=float, default=0.5)

    # Optional: use precomputed predictions instead of running the model
    ap.add_argument("--pred_csv", default=None,
                    help="Optional CSV with columns y_true,y_prob. If provided, model won't run.")
    ap.add_argument("--out_cm", default="confusion_matrix.png")
    ap.add_argument("--out_roc", default="roc_curve.png")
    ap.add_argument("--out_card", default="metrics_card.png")
    args = ap.parse_args()

    if args.pred_csv:
        print(f"[INFO] Loading predictions from {args.pred_csv}")
        y_true, y_prob = load_preds_from_csv(args.pred_csv)
    else:
        print("[INFO] Running model to get test predictions...")
        y_true, y_prob = run_model_and_collect(args)

    y_pred = (y_prob >= args.threshold).astype(int)

    save_confusion_matrix(y_true, y_pred, out_path=args.out_cm)
    save_roc_curve(y_true, y_prob, out_path=args.out_roc)
    save_metrics_card(y_true, y_pred, y_prob, out_path=args.out_card, threshold=args.threshold)


if __name__ == "__main__":
    main()
