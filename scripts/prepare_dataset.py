"""
prepare_dataset.py
==================
Takes the output from extract_frames.py (Dataset/images/ and Dataset/masks/)
and:
  1. Verifies each RGB image has a matching binary mask.
  2. Binarizes the masks (converts near-white pixels to 1, everything else to 0).
  3. Builds a manifest CSV with columns: [image_path, mask_path, split].
  4. Splits into train (70%), val (15%), and test (15%) sets.

The manifest is the input to `train_maskrcnn.py`.

Usage
-----
    python prepare_dataset.py [--dataset_dir PATH]
"""

import argparse
import csv
import random
from pathlib import Path
from PIL import Image
import numpy as np

# ==============================================================================
BASE_DIR      = Path(__file__).parent.parent
DATASET_DIR   = BASE_DIR / "Dataset"
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# TEST_RATIO  = 0.15 (remainder)

# Pixels in the mask with brightness above this threshold are "fire" (=1)
MASK_THRESHOLD = 128   # 0-255 grayscale
# ==============================================================================


def binarize_mask(mask_path: Path, out_path: Path):
    """
    Load a Smokeview-rendered HRRPUV mask (white fire on black background),
    convert to a single-channel binary PNG (255=fire, 0=background).
    """
    img = Image.open(mask_path).convert("L")  # Grayscale
    arr = np.array(img)
    binary = (arr >= MASK_THRESHOLD).astype(np.uint8) * 255
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(binary).save(out_path)


def collect_pairs(dataset_dir: Path):
    """
    Walk Dataset/images/<scenario>/frame_XXXX.png and find the matching
    Dataset/masks/<scenario>/frame_XXXX.png.
    Returns a list of (image_path, mask_path) pairs.
    """
    images_root = dataset_dir / "images"
    masks_root  = dataset_dir / "masks"
    binary_root = dataset_dir / "masks_binary"

    pairs = []
    missing = 0

    for image_path in sorted(images_root.rglob("*.png")):
        # Build corresponding mask path
        relative    = image_path.relative_to(images_root)
        mask_path   = masks_root  / relative
        binary_path = binary_root / relative

        if not mask_path.exists():
            print(f"  [WARN] Missing mask for: {image_path}")
            missing += 1
            continue

        # Binarize the Smokeview mask
        binarize_mask(mask_path, binary_path)
        pairs.append((str(image_path), str(binary_path)))

    print(f"Found {len(pairs)} valid image-mask pairs. ({missing} skipped due to missing masks)")
    return pairs


def write_manifest(pairs: list, output_path: Path):
    """
    Shuffle pairs, assign train/val/test splits, and write a CSV manifest.
    """
    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = (
        [(img, mask, "train") for img, mask in pairs[:n_train]] +
        [(img, mask, "val")   for img, mask in pairs[n_train:n_train + n_val]] +
        [(img, mask, "test")  for img, mask in pairs[n_train + n_val:]]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "mask_path", "split"])
        writer.writerows(splits)

    n_train_out = sum(1 for _, _, s in splits if s == "train")
    n_val_out   = sum(1 for _, _, s in splits if s == "val")
    n_test_out  = sum(1 for _, _, s in splits if s == "test")
    print(f"Manifest written to: {output_path}")
    print(f"  Train: {n_train_out}  |  Val: {n_val_out}  |  Test: {n_test_out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fire segmentation dataset from Smokeview frames")
    parser.add_argument("--dataset_dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    args = parser.parse_args()
    random.seed(args.seed)

    print(f"Dataset dir: {args.dataset_dir}")
    pairs = collect_pairs(args.dataset_dir)
    if not pairs:
        print("ERROR: No paired images found. Run extract_frames.py first.")
        return

    manifest_path = args.dataset_dir / "manifest.csv"
    write_manifest(pairs, manifest_path)
    print("\nDone. Next step: run `train_maskrcnn.py --manifest", manifest_path, "`")


if __name__ == "__main__":
    main()
