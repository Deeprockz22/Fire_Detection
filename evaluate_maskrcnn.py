"""
evaluate_maskrcnn.py
====================
Evaluates the trained Mask R-CNN fire segmentation model on the held-out
test set from manifest.csv.

Metrics reported
----------------
  Pixel-level  : IoU (Jaccard), Precision, Recall, F1, Accuracy
  Detection    : mAP @ IoU=0.5 (instances where mask IoU >= 0.5 count as TP)

Usage
-----
    python evaluate_maskrcnn.py
    python evaluate_maskrcnn.py --checkpoint checkpoints/maskrcnn_fire_best.pt
    python evaluate_maskrcnn.py --score_thr 0.3   # lower confidence threshold
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 2
SCORE_THR    = 0.5   # minimum confidence to accept a predicted mask
MASK_THR     = 0.5   # sigmoid threshold to binarize predicted soft mask
IOU_THR      = 0.5   # IoU threshold used for mAP@0.5

CHECKPOINTS_DIR = Path(r"D:\FDS\Small_project\Fire Detection\checkpoints")
MANIFEST_PATH   = Path(r"D:\FDS\Small_project\Fire Detection\Dataset\manifest.csv")
RESULTS_PATH    = Path(r"D:\FDS\Small_project\Fire Detection\logs\evaluation_results.txt")
# ==============================================================================


# ── Dataset ───────────────────────────────────────────────────────────────────
class FireTestDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str = "test"):
        self.entries = []
        with open(manifest_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.entries.append((row["image_path"], row["mask_path"]))
        print(f"  [{split}] Loaded {len(self.entries)} samples.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path = self.entries[idx]
        image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        gt_binary = (mask >= 128).astype(np.uint8)
        return image, gt_binary, img_path


def collate_fn(batch):
    images, gts, paths = zip(*batch)
    return list(images), list(gts), list(paths)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def merge_predicted_masks(output, h, w, score_thr=SCORE_THR, mask_thr=MASK_THR):
    """
    Combine all predicted fire-class masks into a single binary prediction map.
    Returns a uint8 array of shape (H, W).
    """
    pred = np.zeros((h, w), dtype=np.uint8)
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    masks  = output["masks"].cpu().numpy()   # [N, 1, H, W] float32 soft masks

    for score, label, soft_mask in zip(scores, labels, masks):
        if label == 1 and score >= score_thr:
            binary = (soft_mask[0] >= mask_thr).astype(np.uint8)
            pred = np.maximum(pred, binary)
    return pred


def pixel_metrics(pred: np.ndarray, gt: np.ndarray):
    """Return (iou, precision, recall, f1, pixel_acc) for a single image."""
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    acc       = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return iou, precision, recall, f1, acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=CHECKPOINTS_DIR / "maskrcnn_fire_best.pt")
    parser.add_argument("--manifest",   type=Path, default=MANIFEST_PATH)
    parser.add_argument("--split",      type=str,  default="test")
    parser.add_argument("--score_thr",  type=float, default=SCORE_THR)
    args = parser.parse_args()

    print(f"Device     : {DEVICE}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Split      : {args.split}")
    print(f"Score thr  : {args.score_thr}")
    print("=" * 60)

    ds     = FireTestDataset(args.manifest, split=args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    model = build_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model.eval()

    ious, precs, recs, f1s, accs = [], [], [], [], []
    det_tp, det_fp, det_fn = 0, 0, 0   # for mAP@0.5

    t0 = time.time()
    with torch.no_grad():
        for i, (images, gts, paths) in enumerate(loader):
            image  = images[0].to(DEVICE)
            gt     = gts[0]                         # np (H, W) uint8
            h, w   = gt.shape

            output = model([image])[0]
            pred   = merge_predicted_masks(output, h, w, score_thr=args.score_thr)

            iou, prec, rec, f1, acc = pixel_metrics(pred, gt)
            ious.append(iou); precs.append(prec); recs.append(rec)
            f1s.append(f1);   accs.append(acc)

            # Detection-level mAP@0.5: treat whole frame as one instance
            has_gt   = gt.sum() > 0
            has_pred = pred.sum() > 0

            if has_gt and iou >= IOU_THR:
                det_tp += 1
            elif has_gt and (not has_pred or iou < IOU_THR):
                det_fn += 1
            elif (not has_gt) and has_pred:
                det_fp += 1
            # true negatives (no gt, no pred) are ignored

            if (i + 1) % 20 == 0:
                print(f"  [{i+1:4d}/{len(ds)}]  IoU={iou:.3f}  F1={f1:.3f}  "
                      f"P={prec:.3f}  R={rec:.3f}")

    elapsed = time.time() - t0

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_iou  = float(np.mean(ious))
    mean_prec = float(np.mean(precs))
    mean_rec  = float(np.mean(recs))
    mean_f1   = float(np.mean(f1s))
    mean_acc  = float(np.mean(accs))

    det_precision = det_tp / (det_tp + det_fp + 1e-8)
    det_recall    = det_tp / (det_tp + det_fn + 1e-8)
    det_f1        = 2 * det_precision * det_recall / (det_precision + det_recall + 1e-8)

    lines = [
        "",
        "=" * 60,
        "  EVALUATION RESULTS",
        "=" * 60,
        f"  Samples evaluated   : {len(ds)}",
        f"  Elapsed time        : {elapsed:.1f}s  ({elapsed/len(ds):.2f}s/img)",
        "",
        "  ── Pixel-Level Metrics ──────────────────────────────",
        f"  Mean IoU (Jaccard)  : {mean_iou:.4f}",
        f"  Mean Precision      : {mean_prec:.4f}",
        f"  Mean Recall         : {mean_rec:.4f}",
        f"  Mean F1             : {mean_f1:.4f}",
        f"  Mean Accuracy       : {mean_acc:.4f}",
        "",
        "  ── Detection-Level Metrics (IoU ≥ 0.5) ─────────────",
        f"  TP / FP / FN        : {det_tp} / {det_fp} / {det_fn}",
        f"  Detection Precision : {det_precision:.4f}",
        f"  Detection Recall    : {det_recall:.4f}",
        f"  Detection F1        : {det_f1:.4f}",
        "=" * 60,
    ]

    for line in lines:
        print(line)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines))
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
