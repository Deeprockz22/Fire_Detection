"""
visualize_predictions.py
========================
Produces a grid of side-by-side panels for test set samples:
  Column 1 – Input temperature image
  Column 2 – Ground-truth fire mask (green overlay)
  Column 3 – Predicted fire mask   (red overlay)
  Column 4 – IoU score label

Usage
-----
    python visualize_predictions.py                  # saves 4-up grid to visualizations/
    python visualize_predictions.py --n_samples 20  # show 20 samples
    python visualize_predictions.py --score_thr 0.3
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==============================================================================
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES   = 2
SCORE_THR     = 0.5
MASK_THR      = 0.5
N_SAMPLES     = 16   # number of panels in the output grid

CHECKPOINTS_DIR = Path(r"D:\FDS\Small_project\Fire Detection\checkpoints")
MANIFEST_PATH   = Path(r"D:\FDS\Small_project\Fire Detection\Dataset\manifest.csv")
VIS_DIR         = Path(r"D:\FDS\Small_project\Fire Detection\visualizations")
# ==============================================================================


def build_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


def get_pred_mask(output, h, w, score_thr, mask_thr):
    pred = np.zeros((h, w), dtype=np.uint8)
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    masks  = output["masks"].cpu().numpy()
    for score, label, soft in zip(scores, labels, masks):
        if label == 1 and score >= score_thr:
            pred = np.maximum(pred, (soft[0] >= mask_thr).astype(np.uint8))
    return pred


def overlay(img_rgb: np.ndarray, mask: np.ndarray, color, alpha=0.45):
    """Return RGB image with coloured mask overlay."""
    out = img_rgb.copy().astype(np.float32)
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask > 0,
                                 out[:, :, c] * (1 - alpha) + v * alpha * 255,
                                 out[:, :, c])
    return out.astype(np.uint8)


def iou_score(pred, gt):
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    return float(tp / (tp + fp + fn + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=CHECKPOINTS_DIR / "maskrcnn_fire_best.pt")
    parser.add_argument("--manifest",   type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split",      type=str,  default="test")
    parser.add_argument("--n_samples",  type=int,  default=N_SAMPLES)
    parser.add_argument("--score_thr",  type=float, default=SCORE_THR)
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load test entries ────────────────────────────────────────────────────
    entries = []
    with open(args.manifest, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == args.split:
                entries.append((row["image_path"], row["mask_path"]))

    # Separate fire-present and no-fire to ensure balanced grid
    fire_entries = []
    nofire_entries = []
    for img_p, msk_p in entries:
        gt = np.array(Image.open(msk_p).convert("L"))
        if (gt >= 128).sum() > 0:
            fire_entries.append((img_p, msk_p))
        else:
            nofire_entries.append((img_p, msk_p))

    # 75% fire frames, 25% no-fire frames so grid is informative
    n_fire    = min(int(args.n_samples * 0.75), len(fire_entries))
    n_nofire  = min(args.n_samples - n_fire,     len(nofire_entries))
    selected  = random.sample(fire_entries, n_fire) + random.sample(nofire_entries, n_nofire)
    random.shuffle(selected)
    print(f"Selected {len(selected)} samples  ({n_fire} fire, {n_nofire} no-fire)")

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Run inference & build figure ─────────────────────────────────────────
    n = len(selected)
    ncols = 3
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    fig.patch.set_facecolor("#1a1a2e")

    col_labels = ["Input (Temperature)", "Ground Truth", "Prediction"]
    for ax, lbl in zip(axes[0], col_labels):
        ax.set_title(lbl, color="white", fontsize=11, fontweight="bold", pad=6)

    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for row_i, (img_p, msk_p) in enumerate(selected):
            pil_img = Image.open(img_p).convert("RGB")
            img_np  = np.array(pil_img)
            gt      = (np.array(Image.open(msk_p).convert("L")) >= 128).astype(np.uint8)
            h, w    = gt.shape

            image_tensor = to_tensor(pil_img).to(DEVICE)
            output  = model([image_tensor])[0]
            pred    = get_pred_mask(output, h, w, args.score_thr, MASK_THR)
            iou     = iou_score(pred, gt)

            imgs_to_show = [
                img_np,
                overlay(img_np, gt,   color=(0, 1, 0)),   # green GT
                overlay(img_np, pred, color=(1, 0.2, 0)), # red  Pred
            ]
            border_color = "#00e676" if iou >= 0.5 else ("#ff6b35" if gt.sum() > 0 else "#888")

            for col_i, show_img in enumerate(imgs_to_show):
                ax = axes[row_i, col_i]
                ax.imshow(show_img)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2)
                if col_i == 2:   # annotate the prediction column
                    label = f"IoU={iou:.2f}"
                    color = "#00e676" if iou >= 0.5 else "#ff6b35"
                    ax.set_xlabel(label, color=color, fontsize=9, labelpad=3)

    # Legend
    patches = [
        mpatches.Patch(color=(0, 1, 0, 0.7),      label="Ground Truth"),
        mpatches.Patch(color=(1, 0.2, 0, 0.7),    label="Prediction"),
        mpatches.Patch(color="#00e676",             label="IoU ≥ 0.5 ✓"),
        mpatches.Patch(color="#ff6b35",             label="IoU < 0.5 ✗"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               facecolor="#1a1a2e", labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.6)

    plt.suptitle("Mask R-CNN Fire Segmentation — Test Set",
                 color="white", fontsize=14, fontweight="bold", y=1.002)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = VIS_DIR / "test_predictions_grid.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved grid  → {out_path}")

    # Also save individual panels for the fire frames
    print("Saving individual fire-frame panels...")
    with torch.no_grad():
        for idx, (img_p, msk_p) in enumerate(fire_entries[:8]):
            pil_img = Image.open(img_p).convert("RGB")
            img_np  = np.array(pil_img)
            gt      = (np.array(Image.open(msk_p).convert("L")) >= 128).astype(np.uint8)
            h, w    = gt.shape
            output  = model([to_tensor(pil_img).to(DEVICE)])[0]
            pred    = get_pred_mask(output, h, w, args.score_thr, MASK_THR)
            iou     = iou_score(pred, gt)

            fig2, axs = plt.subplots(1, 3, figsize=(12, 3.5))
            fig2.patch.set_facecolor("#1a1a2e")
            for ax, img_show, title in zip(
                axs,
                [img_np, overlay(img_np, gt, (0,1,0)), overlay(img_np, pred, (1,0.2,0))],
                ["Input", "Ground Truth", f"Prediction  IoU={iou:.3f}"]
            ):
                ax.imshow(img_show)
                ax.set_title(title, color="white", fontsize=10)
                ax.axis("off")
            plt.tight_layout()
            indiv_path = VIS_DIR / f"sample_{idx+1:02d}_iou{iou:.2f}.png"
            plt.savefig(indiv_path, dpi=100, bbox_inches="tight",
                        facecolor=fig2.get_facecolor())
            plt.close()

    print(f"\nAll visualizations saved to: {VIS_DIR}")


if __name__ == "__main__":
    main()
