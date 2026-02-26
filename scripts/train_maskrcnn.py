"""
train_maskrcnn.py
=================
Trains a Mask R-CNN (ResNet-50-FPN backbone) for fire instance segmentation
using the paired FDS/Smokeview synthetic dataset.

The model is initialized with ImageNet weights (transfer learning), then
fine-tuned to detect fire regions as a binary class (background vs. fire).

Usage
-----
    python train_maskrcnn.py --manifest Dataset/manifest.csv
    python train_maskrcnn.py --manifest Dataset/manifest.csv --epochs 30 --batch_size 4

Outputs
-------
    checkpoints/maskrcnn_fire_best.pt   (best validation AP model)
    checkpoints/maskrcnn_fire_last.pt   (last epoch model)
    logs/training_log.csv               (epoch-by-epoch metrics)
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ==============================================================================
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 2        # 1 = fire, 0 = background
NUM_EPOCHS   = 25
BATCH_SIZE   = 2        # Increase if GPU VRAM allows
LR           = 0.001
MOMENTUM     = 0.9
WEIGHT_DECAY = 0.0005
LR_STEP_SIZE = 8        # Decay LR every N epochs
LR_GAMMA     = 0.1
BASE_DIR        = Path(__file__).parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR        = BASE_DIR / "logs"
# ==============================================================================


class FireSegmentationDataset(Dataset):
    """
    Loads paired RGB images and binary fire masks.
    Each item returns:
        image  : FloatTensor [3, H, W]
        target : dict with bounding boxes, labels, and masks for Mask R-CNN
    """

    def __init__(self, manifest_path: Path, split: str, transform=None):
        self.entries   = []
        self.transform = transform

        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.entries.append((row["image_path"], row["mask_path"]))

        print(f"  [{split}] Loaded {len(self.entries)} samples.")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, mask_path = self.entries[idx]

        # Load RGB image
        image = Image.open(img_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image)  # [3, H, W], float32 in [0,1]

        # Load binary mask (255 = fire, 0 = background)
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask)
        binary_mask = (mask_arr >= 128).astype(np.uint8)

        # Mask R-CNN requires one mask per instance. We treat the entire fire
        # region as a single instance in each frame.
        # If there is no fire in this frame, we return an empty target.
        if binary_mask.sum() == 0:
            target = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks":  torch.zeros((0, *binary_mask.shape), dtype=torch.uint8),
            }
            return image_tensor, target

        # Build bounding box from the mask
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        box = torch.tensor([[cmin, rmin, cmax, rmax]], dtype=torch.float32)

        mask_tensor = torch.tensor(binary_mask[None], dtype=torch.uint8)  # [1, H, W]

        target = {
            "boxes":  box,
            "labels": torch.tensor([1], dtype=torch.int64),  # class 1 = fire
            "masks":  mask_tensor,
        }
        return image_tensor, target


def build_model(num_classes: int) -> MaskRCNN:
    """
    Load a pretrained Mask R-CNN and replace the heads with fire-specific heads.
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer     = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    return total_loss / max(len(data_loader), 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Simple validation: average total loss (model stays in train mode)."""
    model.train()  # Mask R-CNN computes losses only in train mode
    total_loss = 0.0
    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict  = model(images, targets)
        losses     = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
    return total_loss / max(len(data_loader), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",    type=Path, default=Path(r"D:\FDS\Small_project\Fire Detection\Dataset\manifest.csv"))
    parser.add_argument("--epochs",      type=int,  default=NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,  default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--resume",      action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Manifest: {args.manifest}")

    # ---- Datasets & Loaders ----
    train_ds = FireSegmentationDataset(args.manifest, split="train")
    val_ds   = FireSegmentationDataset(args.manifest, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # ---- Model ----
    model = build_model(NUM_CLASSES).to(DEVICE)
    print(f"Model: Mask R-CNN ResNet-50-FPN | Classes: {NUM_CLASSES}")

    # ---- Optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.SGD(params, lr=args.lr,
                                 momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    lr_sched   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE,
                                                  gamma=LR_GAMMA)

    # ---- Resume ----
    log_path   = LOGS_DIR / "training_log.csv"
    start_epoch = 1
    best_val    = float("inf")

    if args.resume and (CHECKPOINTS_DIR / "maskrcnn_fire_last.pt").exists():
        model.load_state_dict(torch.load(CHECKPOINTS_DIR / "maskrcnn_fire_last.pt",
                                         map_location=DEVICE))
        # Re-read the log to find the last completed epoch and best val loss
        if log_path.exists():
            import csv as _csv
            rows = list(_csv.DictReader(open(log_path)))
            if rows:
                start_epoch = int(rows[-1]["epoch"]) + 1
                best_val    = min(float(r["val_loss"]) for r in rows)
                # Fast-forward LR scheduler to match
                for _ in range(int(rows[-1]["epoch"])):
                    lr_sched.step()
        print(f"Resumed from epoch {start_epoch-1}. Best val so far: {best_val:.4f}")

    # ---- Training Loop ----
    log_mode = "a" if args.resume else "w"
    with open(log_path, log_mode, newline="") as log_f:
        writer = csv.writer(log_f)
        if not args.resume:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "elapsed_s"])

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
            val_loss   = evaluate(model, val_loader, DEVICE)
            lr_sched.step()

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch [{epoch:02d}/{args.epochs}]  "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"LR: {current_lr:.6f}  "
                  f"Time: {elapsed:.1f}s")

            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}",
                             f"{current_lr:.6f}", f"{elapsed:.1f}"])
            log_f.flush()

            # Save best model
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), CHECKPOINTS_DIR / "maskrcnn_fire_best.pt")
                print(f"  → New best model saved (val_loss={val_loss:.4f})")

            # Always save last
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "maskrcnn_fire_last.pt")

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Best model: {CHECKPOINTS_DIR / 'maskrcnn_fire_best.pt'}")
    print(f"Log file:   {log_path}")


if __name__ == "__main__":
    main()
